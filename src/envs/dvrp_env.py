# from .multiagentenv import MultiAgentEnv
from gym import Env, spaces
import json
import random
import os
import numpy as np
from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
from .utils import dijkstra_paths, fill_cell_im, draw_image
import copy
from .const import *
from .draw import *

# ACTION:
# 0: Wait (Do nothing)
# 1: Return to depot
# 2: Deliver order i (by moving one step towards the respective delivery location)

class DVRPEnv(Env):
    def __init__(self, n_agents = 2, episode_limit=100, expected_orders=15, grid_shape=(10, 10)):
        self.n_agents = n_agents 
        self._episode_length = episode_limit
        self._expected_orders = expected_orders
        self._grid_shape = grid_shape  # size of grid or map
        self.depot_location = (round(self._grid_shape[0] / 2), round(self._grid_shape[1] / 2))  # set at centre of map

        #rewards
        self.invalid_action_penalty = 5
        self.delivery_reward = 2



        # General parameters (changes throughout episode)
        self._vehicle_episode_rewards = [0 for _ in range(self.n_agents)]
        self._total_episode_rewards = 0
        self.rewards = [0] * self.n_agents

        self.n_orders = int(expected_orders)

        self._step_count = 0
        self._clock = 0

        # Vehicle parameters
        self.vehicles_action_history = []
        self.vehicles_pos = {_: (None, None) for _ in range(self.n_agents)}
        self._vehicle_dones = [False] * self.n_agents
        self._successful_delivery = 0
        self._vehicle_mileage = [0] * self.n_agents
        self.vehicle_paths, self.vehicle_path_lengths = dijkstra_paths(self._grid_shape[0], self._grid_shape[1])

        # Limits for observation space variables
        self.vehicle_x_min = 0
        self.vehicle_y_min = 0
        self.order_x_min = -1
        self.order_y_min = -1
        self.clock_min = 0
        self.vehicle_x_max = self._grid_shape[0] - 1
        self.vehicle_y_max = self._grid_shape[1] - 1
        self.order_x_max = self._grid_shape[0] - 1
        self.order_y_max = self._grid_shape[0] - 1
        self.clock_max = self._episode_length

        # Generate homogenous generation of orders throughout episode
        self.order_generation_window = self._episode_length   # time when orders can appear
        self.order_prob = self._expected_orders / self.order_generation_window  # set order probability such that the expected number of orders per day = total_orders
        self.generated_orders = 0

        # Order parameters
        self.orders_pos = [(-1, -1)] * self.n_orders
        self.order_status = [-1] * self.n_orders
        self.order_time = [-1] * self.n_orders
        self._total_appeared_orders = 0

        #Render parameters
        self.icon_av, _ = draw_image('rsz_1rsz_truck.png')
        self.icon_pkg, _ = draw_image('rsz_1pin.png')
        # self.icon_pkg = self.icon_pkg.convert("RGBA")
        self.viewer = None

        # Create observation space
        self._obs_high = np.array([self.vehicle_x_max, self.vehicle_y_max] +
                                  [self.order_x_max, self.order_y_max] * self.n_orders + [self.clock_max])
                                #   [self.order_time_max] * self.n_orders +
                                  # [self.order_distance_max] +
                                   #availability
        self._obs_low = np.array([self.vehicle_x_min, self.vehicle_y_min] +
                                 [self.order_x_min, self.order_y_min] * self.n_orders + [self.clock_min])
                                #  [self.order_status_min] * self.n_orders +
                                #  [self.order_time_min] * self.n_orders +
                                 # [self.order_distance_min] +
                                 
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self.action_max = 1 + self.n_orders + 1

        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.action_max) for _ in range(self.n_agents)])


    def step(self, actions):
        vehicles_action = list(actions)
        self.vehicles_action_history.append(vehicles_action)
        self._step_count += 1
        self._clock += 1
        self.rewards = [0] * self.n_agents

        self.__update_orders()
        self.__update_vehicles_pos(vehicles_action)

        if self._clock >= self._episode_length:
            for i in range(self.n_agents):
                self._vehicle_dones[i] = True

        self._vehicle_episode_rewards = [a + b for a, b in zip(self._vehicle_episode_rewards, self.rewards)]

        return self.get_state(), self.rewards, self._vehicle_dones, {'ongoing': True}

    def __update_orders(self):
        if (self._clock < self.order_generation_window) and (self.generated_orders < self._expected_orders):
            order_x, order_y = self.__generate_order()
            order_i = self.order_status.index(-1)
            # for order_i in range(self.n_orders):
            #     if self.order_status[order_i] == -1:
            self.generated_orders += 1
            self.order_time[order_i] = self._clock
            self.orders_pos[order_i] = (order_x, order_y)
            self.order_status[order_i] = 0

    def get_stats(self):
        return {}

    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        state = self.get_state()
        return state[agent_id]

    def get_obs_size(self):
        return self._obs_high.shape[0]

    def get_state(self):
        _obs = []

        for vehicle_i in range(self.n_agents):
            pos = self.vehicles_pos[vehicle_i]
            flat_list = [pos[0], pos[1]]
            flat_list.extend([x[0] for x in self.orders_pos])
            flat_list.extend([y[1] for y in self.orders_pos])
            flat_list.extend([self._clock])
            _obs.append(flat_list)
        return _obs

    def get_state_size(self):
        return self.get_obs_size()*self.n_agents

    def get_avail_actions(self):
        avail_actions = [0] * self.get_total_actions()
        avail_actions[0] = 1
        avail_actions[1] = 1
        for index, order_i in enumerate(self.order_status):
            if order_i == 0:
                avail_actions[index + 2] = 1
        return avail_actions * self.n_agents
        # return [[1] * self.get_total_actions()] * self.n_agents

    def get_avail_agent_actions(self, agent_id):
        avail_actions = [0] * self.get_total_actions()
        avail_actions[0] = 1
        avail_actions[1] = 1
        for index, order_i in enumerate(self.order_status):
            if order_i == 0:
                avail_actions[index+2] = 1
        return avail_actions

    def get_total_actions(self):
        return self.action_max

    def reset(self):

        # General parameters (changes throughout episode)
        self._vehicle_episode_rewards = [0 for _ in range(self.n_agents)]
        self._total_episode_rewards = 0
        self.rewards = [0] * self.n_agents
        self._step_count = 0
        self._clock = 0

        # Order parameters
        self.orders_pos = [(-1, -1)] * self.n_orders
        self.order_status = [-1] * self.n_orders
        self.order_time = [-1] * self.n_orders
        self._total_appeared_orders = 0
        self.generated_orders = 0

        # Vehicle parameters
        self.vehicles_action_history = []
        self.vehicles_pos = {_: (None, None) for _ in range(self.n_agents)}
        self._vehicle_dones = [False] * self.n_agents
        self._successful_delivery = 0
        self._vehicle_mileage = [0] * self.n_agents
        self.vehicle_paths, self.vehicle_path_lengths = dijkstra_paths(self._grid_shape[0], self._grid_shape[1])

        self.__draw_base_img()
        self.images = [self._base_img]
        self.__init_vehicles()

        return self.get_state()

    def __generate_order(self):
        self._total_appeared_orders += 1
        while True:
            np.random.seed()
            order_x = np.random.randint(self._grid_shape[0])
            order_y = np.random.randint(self._grid_shape[1])
            if (order_x, order_y) != self.depot_location:
                break

        return order_x, order_y

    def __init_vehicles(self):
        for vehicle_i in range(self.n_agents):
            self.vehicles_pos[vehicle_i] = self.depot_location

    def __update_vehicles_pos(self, vehicles_actions):
        for vehicle_i, vehicle_action in enumerate(vehicles_actions):
            self.__update_vehicle_pos(vehicle_i, vehicle_action)

    def __update_vehicle_pos(self, vehicle_i, vehicle_action):
        if vehicle_action == 1:  # return to depot: move one step towards depot location
            self.__vehicle_to_depot(vehicle_i)
        elif 2 <= vehicle_action <= self.action_max:  # deliver order i: move one step towards order i
            order_i = vehicle_action - 2
            if self.order_status[order_i] == 0:
                self.__vehicle_to_order(vehicle_i, order_i)

    def __vehicle_to_order(self, vehicle_i, order_i):
        current_pos = self.vehicles_pos[vehicle_i]
        order_i_pos = self.orders_pos[order_i]
        if current_pos == order_i_pos:
            self.order_status[order_i] = 1 #delivered
        else:
            self._vehicle_mileage[vehicle_i] += 1
            next_pos = (self.vehicle_paths[current_pos][order_i_pos][1][0],
                        self.vehicle_paths[current_pos][order_i_pos][1][1])
            self.vehicles_pos[vehicle_i] = next_pos

    def __vehicle_to_depot(self, vehicle_i):
        current_pos = self.vehicles_pos[vehicle_i]
        if current_pos == self.depot_location:
            pass
        else:
            self._vehicle_mileage[vehicle_i] += 1
            next_pos = (self.vehicle_paths[current_pos][self.depot_location][1][0],
                        self.vehicle_paths[current_pos][self.depot_location][1][1])
            self.vehicles_pos[vehicle_i] = next_pos


    def render(self, mode = 'human', close = False):

        img = copy.copy(self._base_img)

        # Agents
        for agent_i in range(self.n_agents):
            fill_cell_im(img, self.icon_av, self.vehicles_pos[agent_i], cell_size=CELL_SIZE)

        # Orders
        for idx, j in enumerate(self.order_status):
            if j != -1:
                fill_cell_im(img, self.icon_pkg, self.orders_pos[idx], cell_size=CELL_SIZE)
        # img.show()

        # self.images.append(img)
        img = np.asarray(img)
        # img.save('gridworld.jpg', format='JPEG', subsampling=0, quality=100)
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)
        # return self.viewer.render(return_rgb_array = mode=='rgb_array')
        return self.viewer.isopen

    def close(self):

        if self.images is not False:
            self.images[0].save(f'config/envs/results/1.gif', format='GIF',
                                append_images=self.images[1:],
                                save_all=True,
                                duration=len(self.images) / 10, loop=0)

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def seed(self, seed):
        self.seed = seed

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
        


