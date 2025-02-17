import cvxpy as cp
import numpy as np
import math


def formulate_problem(current_state_vec, input_dim : int, f_vec, g_matrix, cost_gradient_function_list, state_cost_function_list, lambda_function_list=None, slack_hyper_param=1., disallowing_slack=False, set_priorities=True, priority_const=1000000., priority_pairs_set=None, use_val_func_for_rhs=False, gamma=1., val_func_list=None, input_multiplier=None):
    assert(len(cost_gradient_function_list) == len(state_cost_function_list))
    if lambda_function_list is not None:
        assert(len(state_cost_function_list) == len(lambda_function_list))
    num_tasks = len(cost_gradient_function_list)

    # Define input and slack variables
    u = cp.Variable(input_dim)
    slack = cp.Variable(num_tasks)

    current_state = current_state_vec

    gradient_vals = [fn(current_state) for fn in cost_gradient_function_list]

    # Define terms used in constraint for executing each task
    drift_vec = np.zeros((num_tasks))
    controllable_matrix = np.zeros((num_tasks, input_dim))
    sigma_output_vec = np.zeros((num_tasks))
    for i in range(num_tasks):
        lie_f = gradient_vals[i].T @ f_vec
        lie_g = gradient_vals[i].T @ g_matrix

        drift_vec[i] = lie_f
        controllable_matrix[i, :] = lie_g

        # Calculate outputs of sigma functions used in RHS of contraint for executing each task
        state_reward_function_val = state_cost_function_list[i](current_state)
        if use_val_func_for_rhs:
            assert(val_func_list is not None)
            assert(len(val_func_list) == num_tasks)
            sigma_output_vec[i] = gamma * val_func_list[i](current_state)
        else:
            sigma_output_vec[i] = math.sqrt((lie_f * lie_f) + (state_reward_function_val * (lie_g @ lie_g.T)))

        # This doesn't seem like the correct thing to do. Fix later.
        if lambda_function_list is not None:
            lambda_function_val = lambda_function_list[i](current_state)
            drift_vec[i] *= (1. / lambda_function_val)
            controllable_matrix[i] *= (1. / lambda_function_val)

    # Define optimization problem
    id1 = np.identity(input_dim)
    id2 = np.identity(num_tasks)
    if input_multiplier is None:
        input_multiplier = 1.
    else:
        slack_hyper_param *= input_multiplier
    objective_fn = cp.Minimize((cp.quad_form(u, id1) * input_multiplier) + (slack_hyper_param * cp.quad_form(slack, id2)))
    if disallowing_slack:
        task_execution_constraint = drift_vec + (controllable_matrix @ u) <= -sigma_output_vec
    else:
        task_execution_constraint = drift_vec + (controllable_matrix @ u) <= -sigma_output_vec + slack

    constraints = [task_execution_constraint]

    if set_priorities and num_tasks > 1:
        assert(not disallowing_slack)
        prioritization_matrix = np.zeros((num_tasks - 1, num_tasks))
        assert(len(priority_pairs_set) == num_tasks - 1) # Allow only num_tasks - 1 pairs to be set (based on papers)
        row = 0
        for higher_priority_task_index, lower_priority_task_index in priority_pairs_set:
            assert((lower_priority_task_index, higher_priority_task_index) not in priority_pairs_set) # Can't have conflicting priorities
            prioritization_matrix[row, higher_priority_task_index] = -1.
            prioritization_matrix[row, lower_priority_task_index] = 1. / priority_const
            row += 1
        priority_constraint = prioritization_matrix @ slack >= 0
        constraints.append(priority_constraint)

    problem = cp.Problem(objective_fn, constraints)

    return problem, u, slack

