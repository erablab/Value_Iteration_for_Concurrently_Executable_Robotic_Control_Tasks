from abc import ABC, abstractmethod
from gymnasium import Env, spaces
import torch
import numpy as np


class AbsControlAffineSys(ABC):
    def __init__(self, state_dim, input_dim, time_step=None, batch_state_cost_func=None, batch_input_cost_func=None, batch_combined_inst_cost_func=None):
        self.state_dim = state_dim
        self.input_dim = input_dim

        self.batch_state_cost_func = batch_state_cost_func
        self.batch_input_cost_func = batch_input_cost_func
        self.batch_combined_inst_cost_func = batch_combined_inst_cost_func
        if (batch_state_cost_func is None and batch_input_cost_func is None and batch_combined_inst_cost_func is not None):
            self.inst_cost_separable = False
            self.inst_cost_exists = True
        elif (batch_state_cost_func is not None and batch_input_cost_func is not None and batch_combined_inst_cost_func is None):
            self.inst_cost_separable = True
            self.inst_cost_exists = True
        elif (batch_state_cost_func is None and batch_input_cost_func is None and batch_combined_inst_cost_func is None):
            self.inst_cost_exists = False
        else:
            assert(False)

        self.time_step = time_step


    def get_state_dim(self):
        return self.state_dim


    def get_input_dim(self):
        return self.input_dim


    def _get_single_vec_input_single_output_from_batch_func(self, batch_func, func_input):
        return batch_func(torch.tensor(np.asarray([func_input]), dtype=torch.float32)).detach().cpu().item()


    def get_batch_inst_cost(self, batch_states : torch.Tensor, batch_inputs):
        batch_size = batch_states.size(dim=0)

        if not self.inst_cost_exists:
            zeros = torch.zeros((batch_size))
            if torch.cuda.is_available():
                zeros = torch.cuda()
            return zeros

        if self.inst_cost_separable:
            batch_state_inst_cost = self.batch_state_cost_func(batch_states)
            batch_input_cost = self.batch_input_cost_func(batch_inputs)
            batch_inst_cost = batch_state_inst_cost + batch_input_cost
        else:
            batch_inst_cost = self.batch_combined_inst_cost_func(batch_states, batch_inputs)

        if self.time_step is not None:
            return batch_inst_cost * self.time_step


    def get_inst_cost(self, state_vec, input_vec):
        if not self.inst_cost_exists:
            return 0

        if self.inst_cost_separable:
            state_inst_cost = self._get_single_vec_input_single_output_from_batch_func(self.batch_state_cost_func, state_vec)
            input_cost = self._get_single_vec_input_single_output_from_batch_func(self.batch_input_cost_func, input_vec)
            inst_cost = state_inst_cost + input_cost
        else:
            state_tensor = torch.tensor(np.asarray([state_vec]), dtype=torch.float32)
            input_tensor = torch.tensor(np.asarray([input_vec]), dtype=torch.float32)
            inst_cost = self.batch_combined_inst_cost_func(state_tensor, input_tensor).detach().cpu().item()

        if self.time_step is not None:
            return inst_cost * self.time_step


    @abstractmethod
    def batch_get_f_vec(self, batch_states : torch.Tensor):
        pass


    @abstractmethod
    def batch_get_g_matrix(self, batch_states : torch.Tensor):
        pass


    def get_f_vec(self, current_state : np.ndarray):
        current_state_tensor = torch.tensor(np.asarray([current_state]), dtype=torch.float32)
        assert(tuple(current_state_tensor.size()) == (1, len(current_state)))
        if torch.cuda.is_available():
            current_state_tensor = current_state_tensor.cuda()
        f_x = self.batch_get_f_vec(current_state_tensor)
        f_x = torch.flatten(f_x)
        f_x = f_x.detach().cpu().numpy()
        return f_x


    def get_g_matrix(self, current_state : np.ndarray):
        current_state_tensor = torch.tensor(np.asarray([current_state]), dtype=torch.float32)
        assert(tuple(current_state_tensor.size()) == (1, len(current_state)))
        if torch.cuda.is_available():
            current_state_tensor = current_state_tensor.cuda()
        g_x = self.batch_get_g_matrix(current_state_tensor)
        assert(list(g_x.size()) == [1, self.get_state_dim(), self.get_input_dim()])
        g_x = torch.squeeze(g_x, dim=0)
        assert(list(g_x.size()) == [self.get_state_dim(), self.get_input_dim()])
        g_x = g_x.detach().cpu().numpy()
        return g_x


    def get_batch_x_dot(self, batch_states : torch.Tensor, batch_inputs : torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available():
            batch_states = batch_states.cuda()
            batch_inputs = batch_inputs.cuda()

        batch_f_x = self.batch_get_f_vec(batch_states)
        batch_g_x = self.batch_get_g_matrix(batch_states)
        return batch_f_x + torch.matmul(batch_g_x, batch_inputs)

    
    def batch_step_in_time(self, batch_states : torch.Tensor, batch_inputs : torch.Tensor, time_step : float) -> torch.Tensor:
        if torch.cuda.is_available():
            batch_states = batch_states.cuda()
            batch_inputs = batch_inputs.cuda()

        return batch_states + ( self.get_batch_x_dot(batch_states, batch_inputs) * time_step )


    def get_x_dot(self, current_state : np.ndarray, input : np.ndarray) -> np.ndarray:
        f_x = self.get_f_vec(current_state)
        g_x = self.get_g_matrix(current_state)

        return f_x + (g_x @ input)


    def step_in_time(self, current_state : np.ndarray, input : np.ndarray, time_step : float) -> np.ndarray:
        return current_state + ( self.get_x_dot(current_state, input) * time_step )
        

class AbsControlAffineEnv(AbsControlAffineSys, Env):
    def __init__(self, state_dim, input_dim, time_step : float, batch_state_cost_func=None, batch_input_cost_func=None, batch_combined_inst_cost_func=None, initial_state_generator=None, episode_lim=None):
        Env.__init__(self)
        AbsControlAffineSys.__init__(self, state_dim, input_dim, time_step, batch_state_cost_func, batch_input_cost_func, batch_combined_inst_cost_func)

        self.current_state = np.zeros((state_dim))

        self.initial_state_generator = initial_state_generator
        self.episode_lim = episode_lim
        self.episode_step_count = 0


    @property
    def observation_space(self):
        return spaces.Box(low=self.get_state_space_lower_bound(), high=self.get_state_space_upper_bound(), shape=(self.get_state_dim(),))


    @property
    def action_space(self):
        return spaces.Box(low=self.get_input_lower_bound(), high=self.get_input_upper_bound(), shape=(self.get_input_dim(),))


    # Concrete implementations must have some way to bound inputs and states for step() to work
    #### INPUT AND STATE BOUNDING ABSTRACT FUNCTIONS #####
    def get_input_lower_bound(self):
        return None


    def get_input_upper_bound(self):
        return None


    def get_state_space_lower_bound(self):
        return None


    def get_state_space_upper_bound(self):
        return None


    def bound_input(self, input):
        return None


    def bound_state(self, state):
        return None


    def step(self, action):
        # Bound input
        bounded_input = self.bound_input(action)
        if bounded_input is not None:
            action = bounded_input
        else:
            assert(self.get_input_lower_bound().shape == (self.get_input_dim(),))
            action = np.maximum(self.get_input_lower_bound(), action)
            assert(self.get_input_upper_bound().shape == (self.get_input_dim(),))
            action = np.minimum(self.get_input_upper_bound(), action)

        inst_cost = self.get_inst_cost(self.current_state, action)

        # Advance state
        self.current_state = self.step_in_time(self.current_state, action, self.time_step)

        # Bound state
        bounded_state = self.bound_state(self.current_state)
        if bounded_state is not None:
            self.current_state = bounded_state
        else:
            assert(self.get_state_space_lower_bound().shape == (self.get_state_dim(),))
            self.current_state = np.maximum(self.get_state_space_lower_bound(), self.current_state)
            assert(self.get_state_space_upper_bound().shape == (self.get_state_dim(),))
            self.current_state = np.minimum(self.get_state_space_upper_bound(), self.current_state)

        # Off-the-shelf RL algorithms may keep going until the end of an episode
        truncated = False
        if self.episode_lim is not None:
            self.episode_step_count += 1
            if self.episode_step_count >= self.episode_lim:
                truncated = True
                self.episode_step_count = 0

        return np.copy(self.current_state), -inst_cost, False, truncated, dict()


    def reset(self, seed=None, options=None): # seed and options parameters are just for following gymnasium API
        if self.initial_state_generator is not None:
            new_initial_state = self.initial_state_generator()
            assert(new_initial_state.shape == self.current_state.shape)
            self.current_state = new_initial_state

            # Bound state
            assert(self.get_state_space_lower_bound().shape == (self.get_state_dim(),))
            self.current_state = np.maximum(self.get_state_space_lower_bound(), self.current_state)
            assert(self.get_state_space_upper_bound().shape == (self.get_state_dim(),))
            self.current_state = np.minimum(self.get_state_space_upper_bound(), self.current_state)

        else:
            rng = np.random.default_rng()
            self.current_state = rng.uniform(low=self.get_state_space_lower_bound(), high=self.get_state_space_upper_bound())

        return np.copy(self.current_state), dict()
