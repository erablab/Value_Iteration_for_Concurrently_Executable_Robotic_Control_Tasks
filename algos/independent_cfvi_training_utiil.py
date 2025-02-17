import torch
import torch.nn as nn
import numpy as np
from .orth_cfvi import get_batch_discrete_time_td_lambda_val_func_est, get_batch_advanced_states
import tqdm
import torch.nn as nn
import torch.optim as optim
import time
import copy


class IndCfviTrainingConfig:
    def __init__(self):
        self.num_iterations = 0

        self.state_dim = 0
        self.input_dim = 0

        self.dataset_of_states = None
        self.dataset_generator = None

        self.batch_f_of_x_func = None
        self.batch_g_of_x_func = None
        self.batch_state_cost_func = None

        self.time_step = 0.01
        self.alpha = 1.
        self.num_steps = 0
        self.gamma = 1.
        self.lamb = 0.95

        self.val_func_est_generation_batch_size = 0
        self.val_func_fitting_batch_size = 0
        self.num_epochs_for_fitting_data = 0

        self.other_task_val_funcs = []
        self.orth_pen_mult_consts = []

        self.val_net = None
        
        self.save_directory = None
        self.save_freq = 10

        self.collect_contraction_hist = True

        self.fitting_learning_rate = 0.01
        self.end_fitting_early = True
        self.fitting_epsilon = 0.01

        self.min_input = None
        self.max_input = None

        self.max_interference_cost = None

        self.use_advanced_states = False
        self.num_states_to_advance = None


    # Verify that the values set in the configuration make sense.
    def verify(self):
        # Verify that all values that should be greater than 0 are greater than 0.
        assert(self.num_iterations > 0)
        assert(self.state_dim > 0)
        assert(self.input_dim > 0)
        assert(self.time_step > 0)
        assert(self.alpha > 0)
        assert(self.num_steps > 0)
        assert(self.gamma > 0)
        assert(self.lamb >= 0) # Still valid if exactly 0 or 1.
        assert(self.val_func_est_generation_batch_size > 0)
        assert(self.val_func_fitting_batch_size > 0)
        assert(self.num_epochs_for_fitting_data > 0)
        assert(self.fitting_learning_rate > 0)
        assert(self.fitting_epsilon > 0)

        # Verify that all values that should be less than or equal to 1 are less than or equal to 1.
        assert(self.alpha <= 1.)
        assert(self.gamma <= 1.)
        assert(self.lamb <= 1.)

        # Verify that at least one of the methods for generating the dataset of states is defined and not both.
        assert(self.dataset_of_states is not None or self.dataset_generator is not None)
        assert(self.dataset_of_states is None or self.dataset_generator is None)

        # Verify that if we advance the states, that the number of states to advance is defined.
        if self.use_advanced_states:
            assert(self.num_states_to_advance is not None)
            assert(self.num_states_to_advance > 0)


class DefaultValNet(nn.Module):
    def __init__(self, num_features, hidden_layer_sizes=[128, 128, 64, 64], normalization_const=None, use_leaky_relu=False):
        super().__init__()
        layers = []
        sizes = [num_features] + hidden_layer_sizes + [1]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                if use_leaky_relu:
                    layers.append(nn.LeakyReLU())
                else:
                    layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.normalization_const = normalization_const

    def forward(self, state):
        if self.normalization_const is not None:
            state = state / self.normalization_const
        return torch.abs(torch.flatten(self.net(state)))


class SoftPlusValNet(nn.Module):
    def __init__(self, num_features, hidden_layer_sizes=[128, 128, 64, 64], normalization_const=None):
        super().__init__()
        layers = []
        sizes = [num_features] + hidden_layer_sizes + [1]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
        self.normalization_const = normalization_const

    def forward(self, state):
        if self.normalization_const is not None:
            state = state / self.normalization_const
        return torch.flatten(self.net(state))


class SimpleQuadraticValNet(nn.Module):
    def __init__(self, num_features, hidden_layer_sizes=[128, 128, 64, 64], normalization_const=None):
        super().__init__()
        layers = []
        sizes = [num_features] + hidden_layer_sizes
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
        self.normalization_const = normalization_const

    def forward(self, state):
        if self.normalization_const is not None:
            state = state / self.normalization_const
        return torch.square(torch.linalg.vector_norm(self.net(state), dim=1))


def get_randomized_batches_from_states_dataset(dataset_of_states, batch_size : int):
    np.random.shuffle(dataset_of_states)
    assert(len(dataset_of_states) % batch_size == 0)
    num_batches = len(dataset_of_states) // batch_size
    batched_dataset = np.zeros(tuple([num_batches] + [batch_size] + list(dataset_of_states.shape[1:])))
    for i in range(num_batches):
        batched_dataset[i] = dataset_of_states[i * batch_size : (i + 1) * batch_size]
    return batched_dataset
    

def rebatch_to_smaller_batches(x_batches, y_batches, old_batch_size, new_batch_size):
    assert((new_batch_size < old_batch_size) and (old_batch_size % new_batch_size == 0))
    assert(len(x_batches) == len(y_batches))

    new_x_batches = []
    new_y_batches = []
    for batch_index in range(len(x_batches)):
        x_batch = x_batches[batch_index]
        y_batch = y_batches[batch_index]

        new_num_batches = old_batch_size // new_batch_size
        for i in range(new_num_batches):
            new_x_batches.append(x_batch[i * new_batch_size : (i + 1) * new_batch_size])
            new_y_batches.append(y_batch[i * new_batch_size : (i + 1) * new_batch_size])

    return new_x_batches, new_y_batches


def get_td_lambda_val_ests_from_batch_of_states(batches_of_states, cfg : IndCfviTrainingConfig):
    batches_of_val_ests = []
    num_batches = len(batches_of_states)
    for i in range(num_batches):
        if num_batches <= 10 or i % (num_batches // 10) == 0:
            print(f"Generated estimates for {i} batches...")
        batch_of_states = batches_of_states[i]
        if torch.cuda.is_available():
            batch_of_states = batch_of_states.cuda()
        batch_of_val_ests = get_batch_discrete_time_td_lambda_val_func_est(cfg.state_dim, cfg.input_dim, cfg.batch_f_of_x_func,
                                                                           cfg.batch_g_of_x_func, cfg.val_net, batch_of_states,
                                                                           cfg.time_step, cfg.batch_state_cost_func, cfg.alpha,
                                                                           cfg.num_steps, cfg.gamma, cfg.lamb, cfg.other_task_val_funcs,
                                                                           cfg.orth_pen_mult_consts, min_input=cfg.min_input, max_input=cfg.max_input,
                                                                           max_interference_cost=cfg.max_interference_cost)
        batch_of_val_ests = batch_of_val_ests.detach().to(torch.float32)
        batches_of_val_ests.append(batch_of_val_ests)
    return batches_of_val_ests


def fit_val_net_to_data(val_net, x_batches, y_batches, num_epochs : int, learning_rate=0.01, end_early=True, epsilon=0.01):
  loss_fn = nn.MSELoss()  # mean square error
  optimizer = optim.Adam(val_net.parameters(), lr=learning_rate)
  optimizer.zero_grad()

  assert(len(x_batches) == len(y_batches))

  val_net.train()

  show_bar_freq = num_epochs // 10

  for epoch in range(num_epochs):
      disable_bar = True
      if num_epochs < 10 or epoch % show_bar_freq == 0:
          disable_bar = False

      total_loss = 0.
      with tqdm.tqdm(list(range(len(x_batches))), unit="batch", mininterval=0, disable=disable_bar) as bar:
          bar.set_description(f"Epoch {epoch}")
          for i in bar:
              # forward pass
              optimizer.zero_grad()
              x_batch = x_batches[i]
              y_batch = y_batches[i]
              y_pred = val_net(x_batch)
              loss = loss_fn(y_pred, y_batch)
              # backward pass
              loss.backward()
              # update weights
              optimizer.step()
              # print progress
              bar.set_postfix(mse=float(loss))
              total_loss += float(loss)

      if end_early and total_loss / len(x_batches) <= epsilon:
          print(f"Finished fitting values in epoch {epoch}. Breaking early.")
          break


def get_contraction_val(old_model, new_model, x_batches):
    old_model.eval()
    new_model.eval()

    num_batches = len(x_batches)

    l1_func = nn.L1Loss()

    total_mae = 0.
    for i in range(num_batches):
        x_batch = x_batches[i]
        mae = l1_func(new_model(x_batch), old_model(x_batch)).detach().cpu().item()
        total_mae += mae

    return total_mae / num_batches


def run_iteration(cfg : IndCfviTrainingConfig, iteration_index : int, dataset_of_states):
    iteration_start_time = time.time()

    cfg.val_net.eval()
    x = get_randomized_batches_from_states_dataset(dataset_of_states, cfg.val_func_est_generation_batch_size)
    x = torch.tensor(x, dtype=torch.float32)
    if torch.cuda.is_available():
        x = x.cuda()

    print(f"Generating value function estimates for iteration {iteration_index}")
    y = get_td_lambda_val_ests_from_batch_of_states(x, cfg)
    if not cfg.val_func_est_generation_batch_size == cfg.val_func_fitting_batch_size:
        x, y = rebatch_to_smaller_batches(x, y, cfg.val_func_est_generation_batch_size, cfg.val_func_fitting_batch_size)

    if cfg.collect_contraction_hist:
        old_model = copy.deepcopy(cfg.val_net)

    print(f"Fitting val net to data for iteration {iteration_index}")
    fit_val_net_to_data(cfg.val_net, x, y, cfg.num_epochs_for_fitting_data, learning_rate=cfg.fitting_learning_rate, end_early=cfg.end_fitting_early, epsilon=cfg.fitting_epsilon)

    if cfg.collect_contraction_hist:
        contraction_val = get_contraction_val(old_model, cfg.val_net, x)

    iteration_time = time.time() - iteration_start_time
    print(f"Iteration {iteration_index} took {iteration_time} seconds")

    if cfg.collect_contraction_hist:
        print(f"Contraction Value: {contraction_val}")
        return contraction_val


def run_multiple_iterations(cfg: IndCfviTrainingConfig):

    if cfg.val_net is None:
        cfg.val_net = DefaultValNet(cfg.state_dim)

    if cfg.collect_contraction_hist:
        contraction_hist = []

    for i in range(cfg.num_iterations):
        if cfg.dataset_generator is not None:
            dataset_of_states = cfg.dataset_generator()
        else:
            dataset_of_states = cfg.dataset_of_states
        
        if cfg.use_advanced_states:
            print(f"Generating advanced states for iteration {i}...")
            generation_start_time = time.time()
            dataset_of_states = torch.tensor(dataset_of_states, dtype=torch.float32)
            if torch.cuda.is_available():
                dataset_of_states = dataset_of_states.cuda()
            assert(cfg.num_states_to_advance is not None)
            dataset_of_states = get_batch_advanced_states(cfg)
            print(f"Finished generating advanced states in {time.time() - generation_start_time} in seconds.")

        res = run_iteration(cfg, i, dataset_of_states)
        if cfg.collect_contraction_hist:
            assert(res is not None)
            contraction_hist.append(res)

        if cfg.save_directory is not None and i % cfg.save_freq == 0:
            torch.save(cfg.val_net.state_dict(), f"{cfg.save_directory}/iteration_{i}.pt")

    if cfg.collect_contraction_hist:
        return cfg.val_net, contraction_hist
    else:
        return cfg.val_net
