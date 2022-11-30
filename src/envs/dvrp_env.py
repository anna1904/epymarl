# from .multiagentenv import MultiAgentEnv
from gym import Env, spaces
import json
import random
import os
import numpy as np
from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace


class DVRPEnv(Env):
    def __init__(self, n_agents = 2, episode_limit=144, expected_orders=15, grid_shape=(10, 10)):
        self.observation_space = [spaces.Box(0, 3, shape=(2,), dtype=int)]
        self.action_space = [spaces.Discrete(4)]
        self.n_agents = n_agents 
        self._episode_length = episode_limit
        self._grid_shape = grid_shape  # size of grid or map
        self.depot_location = (round(self._grid_shape[0] / 2), round(self._grid_shape[1] / 2))  # set at centre of map

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
       


        # Order parameters
        self.orders_pos = {_: [(-1, -1)] * self.n_orders for _ in range(self.n_agents)}
        self.order_status = {_: [-1] * self.n_orders for _ in range(self.n_agents)}
        self.order_time = {_: [-1] * self.n_orders for _ in range(self.n_agents)}

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

        if self._clock >= self._episode_length:
            for i in range(self.n_agents):
                self._vehicle_dones[i] = True

        self._vehicle_episode_rewards = [a + b for a, b in zip(self._vehicle_episode_rewards, self.rewards)]

        return self.__create_state(), self.rewards, self._vehicle_dones, {'ongoing': True}

    def get_stats(self):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        raise NotImplementedError

    def get_obs_size(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        raise NotImplementedError

    def get_total_actions(self):
        raise NotImplementedError

    def reset(self):

        # General parameters (changes throughout episode)
        self._vehicle_episode_rewards = [0 for _ in range(self.n_agents)]
        self._total_episode_rewards = 0
        self.rewards = [0] * self.n_agents
        self._step_count = 0
        self._clock = 0

        # Order parameters
        self.orders_pos = {_: [(-1, -1)] * self.n_orders for _ in range(self.n_agents)}
        self.order_status = {_: [-1] * self.n_orders for _ in range(self.n_agents)}
        self.order_time = {_: [-1] * self.n_orders for _ in range(self.n_agents)}


        self.__init_vehicles()

        # Vehicle parameters
        self.vehicles_action_history = []
        self.vehicles_pos = {_: (None, None) for _ in range(self.n_agents)}
        self._vehicle_dones = [False] * self.n_agents
        self._successful_delivery = 0

        return self.__create_state()

    def __create_state(self):
        _obs = []
        flat_list = []

        for vehicle_i in range(self.n_agents):
            pos = self.vehicles_pos[vehicle_i]
            flat_list = [pos[0], pos[1]]
            flat_list.extend([x[0] for x in self.orders_pos[vehicle_i]])
            flat_list.extend([y[0] for y in self.orders_pos[vehicle_i]])
            # flat_list.extend(self.order_status[vehicle_i])
            # flat_list.extend(self.order_time[vehicle_i])
            # flat_list.extend(self.order_distance)
            flat_list.extend([self._clock])
            _obs.append(flat_list)

        return _obs

    def __generate_order(self):
        self._total_appeared_orders += 1
        while True:
            np.random.seed()
            order_x = np.random.randint(self._grid_shape[0])  # todo for some reason it keeps on genearting (5,1)
            order_y = np.random.randint(self._grid_shape[1])
            if (order_x, order_y) != self.depot_location:
                break

        return order_x, order_y

    def __init_vehicles(self):
        for vehicle_i in range(self.n_agents):
            self.vehicles_pos[vehicle_i] = self.depot_location

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self, seed):
        self.seed = seed

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
        


