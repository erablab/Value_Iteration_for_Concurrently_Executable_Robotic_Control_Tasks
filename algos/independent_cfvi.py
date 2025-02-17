import torch


def get_batch_grads(val_func, batch_x, unsqueeze=False):
    # Compute the "batch" Jacobian
    def val_func_outputs_summed(batch_x):
        return torch.sum(val_func(batch_x))
    batch_grad = torch.autograd.functional.jacobian(val_func_outputs_summed, batch_x)
    if unsqueeze:
        batch_grad = torch.unsqueeze(batch_grad, dim=-1)
    return batch_grad


def get_batch_optimal_input_from_val_func(state_dim : int, input_dim : int, batch_g_of_x, val_func, batch_x, alpha=1., other_task_val_funcs=[], orth_pen_mult_consts=[]):
    batch_size = batch_x.size(dim=0)

    # Create batch identity matrix (component that penalizes input in general)
    ident = torch.eye(input_dim)
    ident = ident.reshape((1, input_dim, input_dim))
    ident = ident.repeat(batch_size, 1, 1)
    if torch.cuda.is_available():
        ident = ident.cuda()

    # Calculate the gradients of the other value functions
    other_task_batch_grads = []
    for v in other_task_val_funcs:
        other_task_batch_grads.append(get_batch_grads(v, batch_x, unsqueeze=True))
    
    # Calculate the summation of the symmetric squares of the gradients multiplied by the corresponding constant
    assert(len(other_task_batch_grads) == len(orth_pen_mult_consts))
    batch_other_tasks_squared_sum = torch.zeros((batch_size, state_dim, state_dim))
    if torch.cuda.is_available():
        batch_other_tasks_squared_sum = batch_other_tasks_squared_sum.cuda()
    for i, grad in enumerate(other_task_batch_grads):
        batch_other_tasks_squared_sum = batch_other_tasks_squared_sum + (torch.bmm(grad, torch.transpose(grad, 1, 2)) * orth_pen_mult_consts[i]) 

    batch_g_matrix = batch_g_of_x(batch_x)

    # Calculate inverse of (alpha * I + g(x)^T(sum of squares of gradients)g(x))
    batch_inv = torch.linalg.inv((alpha * ident) + torch.bmm(torch.transpose(batch_g_matrix, 1, 2), torch.bmm(batch_other_tasks_squared_sum, batch_g_matrix)))

    # Calculate the gradient for the current task being trained
    batch_grads = get_batch_grads(val_func, batch_x, unsqueeze=True)

    optimal_input = (-1./2) * torch.bmm(batch_inv, torch.bmm(torch.transpose(batch_g_matrix, 1, 2), batch_grads))
    optimal_input = torch.squeeze(optimal_input, dim=-1)
    return optimal_input


def get_batch_discrete_time_n_step_val_func_est(state_dim : int, input_dim : int, batch_f_of_x, batch_g_of_x, val_func, batch_initial_x, time_step, batch_state_cost_func, alpha=1., n=1, gamma=1., other_task_val_funcs=[], orth_pen_mult_consts=[]):
    assert(n >= 1)
    assert(alpha > 0)
    assert(gamma > 0. and gamma <= 1.)

    batch_size = batch_initial_x.size(dim=0)

    batch_monte_carlo_cost = torch.zeros(batch_size)
    if torch.cuda.is_available():
        batch_monte_carlo_cost = batch_monte_carlo_cost.cuda()
    batch_current_x = batch_initial_x
    for i in range(n):
        batch_optimal_input = get_batch_optimal_input_from_val_func(state_dim, input_dim, batch_g_of_x, val_func, batch_current_x, alpha, other_task_val_funcs, orth_pen_mult_consts)

        # Calculate the instantaneous cost from the current state and from norm of the input
        current_state_cost = batch_state_cost_func(batch_current_x)
        total_batch_step_cost = current_state_cost + (torch.square(torch.linalg.vector_norm(batch_optimal_input, dim=1)) * alpha)
        assert(list(total_batch_step_cost.size()) == [batch_size])

        # Calculate the instantataneous cost incurred from interfering with other tasks
        batch_g_matrix = batch_g_of_x(batch_current_x)
        for j,v in enumerate(other_task_val_funcs):
            batch_grads = get_batch_grads(v, batch_current_x, unsqueeze=True)
            batch_grads = torch.transpose(batch_grads, 1, 2)
            lie_g_v = torch.bmm(batch_grads, batch_g_matrix)
            assert(list(lie_g_v.size()) == [batch_size, 1, input_dim])
            batch_optimal_input_unsqueezed = torch.unsqueeze(batch_optimal_input, dim=-1)
            interference_cost = torch.bmm(lie_g_v, batch_optimal_input_unsqueezed)
            interference_cost = torch.square(torch.flatten(interference_cost))
            assert(list(interference_cost.size()) == [batch_size])
            total_batch_step_cost = total_batch_step_cost + (interference_cost * orth_pen_mult_consts[j])

        batch_monte_carlo_cost = batch_monte_carlo_cost + (total_batch_step_cost * (gamma**i) * time_step)

        # Calculate batch derivative of x
        unsqueezed_batch_optimal_input = torch.unsqueeze(batch_optimal_input, dim=-1)
        batch_forced_resp = torch.bmm(batch_g_of_x(batch_current_x), unsqueezed_batch_optimal_input)
        batch_forced_resp = torch.squeeze(batch_forced_resp, dim=-1)
        batch_current_x_deriv = batch_f_of_x(batch_current_x) + batch_forced_resp

        # Update the current x
        batch_current_x = batch_current_x + (batch_current_x_deriv * time_step)

    batch_inf_horizon_est = val_func(batch_current_x) * (gamma**n)

    return batch_monte_carlo_cost + batch_inf_horizon_est

        
def get_batch_discrete_time_td_lambda_val_func_est(state_dim : int, input_dim : int, batch_f_of_x, batch_g_of_x, val_func, batch_initial_x, time_step, batch_state_cost_func, alpha=1., n=1, gamma=0.99, lamb=0.95, other_task_val_funcs=[], orth_pen_mult_consts=[], min_input=None, max_input=None, max_interference_cost=1000.):
    assert(n >= 1)
    assert(alpha > 0)
    assert(gamma > 0. and gamma <= 1.)
    assert(lamb > 0. and lamb < 1.)

    batch_size = batch_initial_x.size(dim=0)

    batch_monte_carlo_cost = torch.zeros(batch_size)
    batch_total_est = torch.zeros(batch_size)
    if torch.cuda.is_available():
        batch_monte_carlo_cost = batch_monte_carlo_cost.cuda()
        batch_total_est = batch_total_est.cuda()
    batch_current_x = batch_initial_x

    for i in range(n):
        batch_optimal_input = get_batch_optimal_input_from_val_func(state_dim, input_dim, batch_g_of_x, val_func, batch_current_x, alpha, other_task_val_funcs, orth_pen_mult_consts)
        if min_input is not None or max_input is not None:
            batch_optimal_input = torch.clamp(batch_optimal_input, min=min_input, max=max_input)
        assert(list(batch_optimal_input.size()) == [batch_size, input_dim])

        # Calculate the instantaneous state cost
        batch_current_state_cost = batch_state_cost_func(batch_current_x)

        # Calculate the instantaneous cost from control effort
        batch_control_effort_cost = torch.square(torch.linalg.vector_norm(batch_optimal_input, dim=1)) * alpha

        # Calculate the instantataneous cost incurred from interfering with other tasks
        batch_g_matrix = batch_g_of_x(batch_current_x)
        total_interference_cost = torch.zeros(batch_size)
        if torch.cuda.is_available():
            total_interference_cost = total_interference_cost.cuda()
        for j,v in enumerate(other_task_val_funcs):
            batch_grads = get_batch_grads(v, batch_current_x, unsqueeze=True)
            batch_grads = torch.transpose(batch_grads, 1, 2)
            lie_g_v = torch.bmm(batch_grads, batch_g_matrix)
            assert(list(lie_g_v.size()) == [batch_size, 1, input_dim])
            batch_optimal_input_unsqueezed = torch.unsqueeze(batch_optimal_input, dim=-1)
            interference_cost = torch.bmm(lie_g_v, batch_optimal_input_unsqueezed)
            interference_cost = torch.square(torch.flatten(interference_cost)) * orth_pen_mult_consts[j]
            assert(list(interference_cost.size()) == [batch_size])
            total_interference_cost = total_interference_cost + interference_cost

        if max_interference_cost is not None:
            assert(max_interference_cost > 0.)
            total_interference_cost = torch.clamp(total_interference_cost, max=max_interference_cost)

        # Update the cumulative weighted sum of costs
        batch_total_inst_cost = batch_current_state_cost + batch_control_effort_cost + total_interference_cost
        batch_monte_carlo_cost = batch_monte_carlo_cost + (batch_total_inst_cost * (gamma ** i) * time_step)

        # Calculate batch derivative of x
        unsqueezed_batch_optimal_input = torch.unsqueeze(batch_optimal_input, dim=-1)
        batch_forced_resp = torch.bmm(batch_g_of_x(batch_current_x), unsqueezed_batch_optimal_input)
        batch_forced_resp = torch.squeeze(batch_forced_resp, dim=-1)
        batch_current_x_deriv = batch_f_of_x(batch_current_x) + batch_forced_resp

        # Update the current x
        batch_current_x = batch_current_x + (batch_current_x_deriv * time_step)

        # Calculate the n-step estimate
        batch_inf_horizon_est = val_func(batch_current_x) * (gamma ** (i + 1))
        batch_n_step_est = batch_monte_carlo_cost + batch_inf_horizon_est

        # Add the n-step estimate to the TD(lambda) estimate
        if i == (n - 1):
            batch_total_est = batch_total_est + ((lamb ** i) * batch_n_step_est)
        else:
            batch_total_est = batch_total_est + ((1 - lamb) * (lamb ** i) * batch_n_step_est)

    return batch_total_est


def get_batch_advanced_states(state_dim : int, input_dim : int, batch_f_of_x, batch_g_of_x, val_func, batch_initial_x, time_step, alpha=1., n=1, other_task_val_funcs=[], orth_pen_mult_consts=[], min_input=None, max_input=None, flatten_set_of_states=True):
    assert(n >= 1)
    assert(alpha > 0)

    identity_func = torch.nn.Identity()

    batch_size = batch_initial_x.size(dim=0)
    batch_current_x = batch_initial_x
    batch_x_hist = [identity_func(batch_current_x)]

    for i in range(n):
        batch_optimal_input = get_batch_optimal_input_from_val_func(state_dim, input_dim, batch_g_of_x, val_func, batch_current_x, alpha, other_task_val_funcs, orth_pen_mult_consts)
        if min_input is not None or max_input is not None:
            batch_optimal_input = torch.clamp(batch_optimal_input, min=min_input, max=max_input)
        assert(list(batch_optimal_input.size()) == [batch_size, input_dim])

        # Calculate batch derivative of x
        unsqueezed_batch_optimal_input = torch.unsqueeze(batch_optimal_input, dim=-1)
        batch_forced_resp = torch.bmm(batch_g_of_x(batch_current_x), unsqueezed_batch_optimal_input)
        batch_forced_resp = torch.squeeze(batch_forced_resp, dim=-1)
        batch_current_x_deriv = batch_f_of_x(batch_current_x) + batch_forced_resp

        # Update the current x
        batch_current_x = batch_current_x + (batch_current_x_deriv * time_step)

        batch_x_hist.append(identity_func(batch_current_x))

    if flatten_set_of_states:
        return torch.cat(batch_x_hist, dim=0).detach().cpu().numpy()
    else:
        return torch.stack(batch_x_hist, dim=0).detach().cpu().numpy()
