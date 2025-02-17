from abc import abstractmethod
from .abstract_control_affine_sys import AbsControlAffineEnv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import PIL
import torch


class SimpleMultiRobot(AbsControlAffineEnv):
    def __init__(self, num_robots : int, state_dim_per_robot : int, input_dim_per_robot : int, time_step : float, min_x : float, max_x : float, min_y : float, max_y : float, batch_state_cost_func=None, batch_input_cost_func=None, batch_combined_inst_cost_func=None, initial_state_generator=None, episode_lim=None, robot_colours=None, render_overlay_func=None):
        self.num_robots = num_robots
        AbsControlAffineEnv.__init__(self, self.num_robots * state_dim_per_robot, self.num_robots * input_dim_per_robot, time_step, batch_state_cost_func, batch_input_cost_func, batch_combined_inst_cost_func, initial_state_generator, episode_lim)
        assert(min_x < max_x)
        assert(min_y < max_y)
        self.__min_x = min_x
        self.__max_x = max_x
        self.__min_y = min_y
        self.__max_y = max_y
        self.robot_colours = robot_colours
        self.render_overlay_func = render_overlay_func


    @property
    def min_x(self):
        return self.__min_x


    @property
    def max_x(self):
        return self.__max_x


    @property
    def min_y(self):
        return self.__min_y


    @property
    def max_y(self):
        return self.__max_y


    def render(self):
        position_vec = self.get_robot_position_vec_from_state(self.current_state)
        assert(position_vec.shape == (self.num_robots * 2,))
        robot_positions = [(position_vec[i * 2], position_vec[i * 2 + 1]) for i in range(self.num_robots)]

        plt.ioff()

        fig, ax = plt.subplots()

        ax.set_xlim(left=self.min_x, right=self.max_x)
        ax.set_ylim(bottom=self.min_y, top=self.max_y)

        maj_pos = ticker.MultipleLocator(1.)
        min_pos = ticker.MultipleLocator(0.25)
        ax.xaxis.set(major_locator=maj_pos, minor_locator=min_pos)
        ax.yaxis.set(major_locator=maj_pos, minor_locator=min_pos)
        ax.tick_params(axis='both', which='minor', length=0)   # remove minor tick lines
        ax.grid(which='major', alpha=0.5)
        ax.grid(which='minor', alpha=0.5, linestyle='--')
        ax.set_axisbelow(True)

        for i in range(self.num_robots):
            x = robot_positions[i][0]
            y = robot_positions[i][1]
            if self.robot_colours is not None:
                robot_colour = self.robot_colours[i]
            else:
                robot_colour = 'g'
            RADIUS = 0.1
            ax.add_patch(plt.Circle((x, y), RADIUS, color=robot_colour, zorder=3))

        if self.render_overlay_func is not None:
            self.render_overlay_func(ax)

        fig.canvas.draw()
        img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        plt.close()
        return img


    @abstractmethod
    def get_robot_position_vec_from_state(self, state_vec : np.ndarray) -> np.ndarray:
        pass


class SingleIntegratorMultiRobot(SimpleMultiRobot):
    def __init__(self, num_robots : int, time_step : float,  min_x : float, max_x : float, min_y : float, max_y : float, max_x_speed : float, max_y_speed : float, batch_state_cost_func=None, batch_input_cost_func=None, batch_combined_inst_cost_func=None, initial_state_generator=None, episode_lim=None, robot_colours=None, render_overlay_func=None):
        super().__init__(num_robots, 2, 2, time_step, min_x, max_x, min_y, max_y, batch_state_cost_func, batch_input_cost_func, batch_combined_inst_cost_func, initial_state_generator, episode_lim, robot_colours, render_overlay_func)
        assert(max_x_speed > 0)
        assert(max_y_speed > 0)
        self.max_x_speed = max_x_speed
        self.max_y_speed = max_y_speed


    # Implement functions needed for AbsControlAffineSys
    ########################################################
    def batch_get_f_vec(self, batch_states : torch.Tensor):
        batch_size  = batch_states.size(dim=0)
        state_dim = self.get_state_dim()
        f_vec = torch.zeros((batch_size, state_dim))
        if torch.cuda.is_available():
            f_vec = f_vec.cuda()
        return f_vec


    def batch_get_g_matrix(self, batch_states : torch.Tensor):
        batch_size  = batch_states.size(dim=0)
        state_dim = self.get_state_dim()
        input_dim = self.get_input_dim()
        assert(state_dim == input_dim)
        identity_matrix = torch.eye(state_dim)
        identity_matrix = torch.unsqueeze(identity_matrix, dim=0)
        batch_identity_matrix = identity_matrix.repeat(batch_size, 1, 1)
        if torch.cuda.is_available():
            batch_identity_matrix = batch_identity_matrix.cuda()
        return batch_identity_matrix


    # Implement functions needed for AbsControlAffineEnv
    ########################################################
    def get_input_lower_bound(self):
        lower_bound = np.zeros(self.num_robots * 2)
        for i in range(self.num_robots):
            lower_bound[i * 2] = -self.max_x_speed
            lower_bound[i * 2 + 1] = -self.max_y_speed
        return lower_bound
        

    def get_input_upper_bound(self):
        upper_bound = np.zeros(self.num_robots * 2)
        for i in range(self.num_robots):
            upper_bound[i * 2] = self.max_x_speed
            upper_bound[i * 2 + 1] = self.max_y_speed
        return upper_bound


    def get_state_space_lower_bound(self):
        lower_bound = np.zeros(self.num_robots * 2)
        for i in range(self.num_robots):
            lower_bound[i * 2] = self.min_x
            lower_bound[i * 2 + 1] = self.min_y
        return lower_bound


    def get_state_space_upper_bound(self):
        upper_bound = np.zeros(self.num_robots * 2)
        for i in range(self.num_robots):
            upper_bound[i * 2] = self.max_x
            upper_bound[i * 2 + 1] = self.max_y
        return upper_bound


    # Implement functions needed for SimpleMultiRobot
    ########################################################
    def get_robot_position_vec_from_state(self, state_vec: np.ndarray) -> np.ndarray:
        return state_vec


class DoubleIntegratorMultiRobot(SimpleMultiRobot):
    def __init__(self, num_robots : int, time_step : float,  min_x : float, max_x : float, min_y : float, max_y : float, max_x_speed : float, max_y_speed : float, max_x_accel : float, max_y_accel : float, batch_state_cost_func=None, batch_input_cost_func=None, batch_combined_inst_cost_func=None, initial_state_generator=None, episode_lim=None, robot_colours=None, render_overlay_func=None):
        super().__init__(num_robots, 4, 2, time_step, min_x, max_x, min_y, max_y, batch_state_cost_func, batch_input_cost_func, batch_combined_inst_cost_func, initial_state_generator, episode_lim, robot_colours, render_overlay_func)
        assert(max_x_speed > 0)
        assert(max_x_accel > 0)
        assert(max_y_speed > 0)
        assert(max_y_accel > 0)
        self.max_x_speed = max_x_speed
        self.max_x_accel = max_x_accel
        self.max_y_speed = max_y_speed
        self.max_y_accel = max_y_accel


    # Implement functions needed for AbsControlAffineSys
    ########################################################
    def batch_get_f_vec(self, batch_states : torch.Tensor):
        # Create projection matrix for a single multi-robot system
        projection_matrix = np.zeros((self.get_state_dim(), self.get_state_dim()))
        assert(self.get_state_dim() == self.num_robots * 4)
        for i in range(self.num_robots):
            projection_matrix[i * 4, i * 4 + 2] = 1.
            projection_matrix[i * 4 + 1, i * 4 + 3] = 1.

        # Convert to a tensor and repeat for the appropriate batch size
        projection_tensor = torch.tensor(projection_matrix, dtype=torch.float32)
        if torch.cuda.is_available():
            projection_tensor = projection_tensor.cuda()
        batch_size = batch_states.size(dim=0)
        batch_projection_tensor = projection_tensor.repeat(batch_size, 1, 1)
        f_x = torch.bmm(batch_projection_tensor, torch.unsqueeze(batch_states, dim=-1))
        return f_x


    def batch_get_g_matrix(self, batch_states : torch.Tensor):
        # Create g(x) matrix for a single multi-robot system
        g_x = np.zeros((self.get_state_dim(), self.get_input_dim()))
        for i in range(self.num_robots):
            g_x[i * 4 + 2, i * 2] = 1.
            g_x[i * 4 + 3, i * 2 + 1] = 1.

        # Convert to a tensor and repeat for the appropriate batch size
        g_x_tensor = torch.tensor(g_x, dtype=torch.float32)
        if torch.cuda.is_available():
            g_x_tensor = g_x_tensor.cuda()
        batch_size  = batch_states.size(dim=0)
        batch_g_x_tensor = g_x_tensor.repeat(batch_size, 1, 1)
        return batch_g_x_tensor


    # Implement functions needed for AbsControlAffineEnv
    ########################################################
    def get_input_lower_bound(self):
        lower_bound = np.zeros(self.num_robots * 2)
        for i in range(self.num_robots):
            lower_bound[i * 2] = -self.max_x_accel
            lower_bound[i * 2 + 1] = -self.max_y_accel
        return lower_bound
        

    def get_input_upper_bound(self):
        upper_bound = np.zeros(self.num_robots * 2)
        for i in range(self.num_robots):
            upper_bound[i * 2] = self.max_x_accel
            upper_bound[i * 2 + 1] = self.max_y_accel
        return upper_bound


    def get_state_space_lower_bound(self):
        lower_bound = np.zeros(self.num_robots * 4)
        for i in range(self.num_robots):
            lower_bound[i * 4] = self.min_x
            lower_bound[i * 4 + 1] = self.min_y
            lower_bound[i * 4 + 2] = -self.max_x_speed
            lower_bound[i * 4 + 3] = -self.max_y_speed
        return lower_bound


    def get_state_space_upper_bound(self):
        upper_bound = np.zeros(self.num_robots * 4)
        for i in range(self.num_robots):
            upper_bound[i * 4] = self.max_x
            upper_bound[i * 4 + 1] = self.max_y
            upper_bound[i * 4 + 2] = self.max_x_speed
            upper_bound[i * 4 + 3] = self.max_y_speed
        return upper_bound


    def get_robot_position_vec_from_state(self, state_vec: np.ndarray) -> np.ndarray:
        assert((self.num_robots * 4,) == state_vec.shape)
        position_vec = np.zeros((self.num_robots * 2))
        for i in range(self.num_robots):
            position_vec[i * 2] = state_vec[i * 4]
            position_vec[i * 2 + 1] = state_vec[i * 4 + 1]
        return position_vec


def get_double_integrator_zero_vel_initial_state_generator_func(num_robots, min_x, max_x, min_y, max_y):
    def generator_func():
        rng = np.random.default_rng()
        initial_state_vec = np.zeros((num_robots * 4))
        for i in range(num_robots):
            initial_state_vec[i * 4 : i * 4 + 2] = rng.uniform(low=[min_x, min_y], high=[max_x, max_y])
        return initial_state_vec
    return generator_func
