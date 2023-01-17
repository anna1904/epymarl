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
from .sim_annel import *
from .cheapest_insertion import *

# ACTION:
# 0: Wait (Do nothing)
# 1 Accept the order
# 2: Return to depot
# 3: Deliver order i (by moving one step towards the respective delivery location)

#ORDER STATUS:
# -1: InACtive
# 0: Available
# 1: Accepted by this agent
# 2: Accepted by another agent
#3: Delivered
#4: Rejected

class DVRPEnv(Env):
    def __init__(self, n_agents = 2, episode_limit=50, expected_orders=4, grid_shape=(10, 10), generated_points = 20):
        self.n_agents = n_agents 
        self._episode_length = episode_limit
        self._expected_orders = expected_orders
        self._generated_points = generated_points
        self._grid_shape = grid_shape  # size of grid or map
        self.depot_location = (round(self._grid_shape[0] / 2), round(self._grid_shape[1] / 2))  # set at centre of map
        self.points_locations = [(np.random.randint(grid_shape[0]), np.random.randint(grid_shape[1])) for _ in range(generated_points)]

        #rewards
        self.invalid_action_penalty = 5
        self.delivery_reward = 2
        self.accept_reward = 1

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
        self._total_accepted_orders = 0
        self._total_delivered_orders = 0
        self._total_rejected_orders = 0
        self._vehicle_mileage = [0] * self.n_agents
        self.vehicle_paths, self.vehicle_path_lengths = dijkstra_paths(self._grid_shape[0], self._grid_shape[1])
        self.vehicles_moving = [False for _ in range(self.n_agents)]

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
        self.current_order_id = -1

        # Order parameters
        self.orders_pos = [(-1, -1)] * self.n_orders
        self.order_status = [-1] * self.n_orders
        self.order_time = [-1] * self.n_orders
        self.order_vehicle = [-1] * self.n_orders #number of agent which accept an order
        self._total_appeared_orders = 0

        #Render parameters
        self.icon_av, _ = draw_image('rsz_1rsz_truck.png')
        self.icon_pkg, _ = draw_image('rsz_1pin.png')
        self.icon_delivered, _ = draw_image('rsz_delivered.png')
        self.viewer = None
        self.images = False

        # Create observation space
        self._obs_high = np.array([self.vehicle_x_max, self.vehicle_y_max] +
                                  [self.vehicle_x_max, self.vehicle_y_max] * self.n_orders +
                                  [self.clock_max] +
                                  [4] * self.n_orders)
                                  # [self.vehicle_x_max, self.vehicle_y_max])

        self._obs_low = np.array([self.vehicle_x_min, self.vehicle_y_min] +
                                  [self.vehicle_x_min, self.vehicle_y_min] * self.n_orders +
                                  [self.clock_min] +
                                  [-1] * self.n_orders)
                                 # [self.vehicle_x_min, self.vehicle_y_max] #the last is the location to which agent is moving
                                 
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self.action_max = 1 + 1 + 1 + self.n_orders  #do nothing, accept, depot, move to order


        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.action_max) for _ in range(self.n_agents)])


    def step(self, actions):
        vehicles_action = list(actions)
        self.vehicles_action_history.append(vehicles_action)
        self._step_count += 1
        self._clock += 1
        self.rewards = [0] * self.n_agents

        #no order available but vehicle drive there

        self.__generate_orders()
        self.__update_action_1(vehicles_action)
        self.__update_vehicles_pos(vehicles_action) #move either to depot, location
        self.__update_rewards(vehicles_action) #give penalty a)to select unavailable order b) to stay in depot for next action

        if self._clock >= self._episode_length or self.order_status.count(2) == self._expected_orders:
            for i in range(self.n_agents):
                self._vehicle_dones[i] = True

        self._vehicle_episode_rewards = [a + b for a, b in zip(self._vehicle_episode_rewards, self.rewards)]

        return self.get_state(), self.rewards, self._vehicle_dones, {'ongoing': True}

    def __generate_orders(self):
        if (self._clock < self.order_generation_window) and (self.generated_orders < self._expected_orders) and (self._clock % 10 == 1):
            order_x, order_y = self.__select_order()
            self.current_order_id = self.generated_orders

            if self.order_status[self.current_order_id - 1] == 0:
                self.order_status[self.current_order_id - 1] = 4

            order_i = self.current_order_id
            self.generated_orders += 1
            self.order_time[order_i] = self._clock
            self.orders_pos[order_i] = (order_x, order_y)
            self.order_status[order_i] = 0

    def __update_rewards(self, vehicles_action):
        if self.order_status.count(2) == self._expected_orders:
            self.rewards = [reward + 20 for reward in self.rewards]
            return

        for vehicle_i, vehicle_action in enumerate(vehicles_action):
            if 3 <= vehicle_action <= self.action_max:
                order_i = vehicle_action - 3
                if self.order_status[order_i] == 1 and \
                        self.vehicles_pos[vehicle_i] == self.orders_pos[order_i] and self.order_vehicle[order_i] == vehicle_i:
                    self._total_delivered_orders += 1
                    self.rewards[vehicle_i] += self.delivery_reward  # give reward for successful delivery
                    self._successful_delivery += 1
                elif self.order_status[order_i] == -1:
                    self.rewards[
                        vehicle_i] -= self.invalid_action_penalty  # give large penalty to vehicle for invalid action
            if vehicle_action == 0 and self.vehicles_action_history[self._step_count-1][vehicle_i] == 0:
                self.rewards[
                    vehicle_i] -= self.invalid_action_penalty  # give large penalty to vehicle for invalid action



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
            flat_list.extend(self.get_order_status(vehicle_i))
            _obs.append(flat_list)
        return _obs
    def get_order_status(self, vehicle_i):
        order_status_agent = copy.deepcopy(self.order_status)
        for order in range(self.n_orders):
            if self.order_status[order] == 1 and self.order_vehicle[order] == vehicle_i:
                order_status_agent[order] == 1
            else:
                order_status_agent[order] == 2
        return order_status_agent

    def get_state_size(self):
        return self.get_obs_size()*self.n_agents

    def get_avail_agent_actions(self, agent_id):
        avail_actions = np.array([0] * self.get_total_actions())
        avail_actions[0:2] = 1
        for index, order_i in enumerate(self.order_status):
            if order_i == 1 and self.order_vehicle[index] == agent_id:
                avail_actions[index] = 1
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
        self.current_order_id = -1

        # Vehicle parameters
        self.vehicles_action_history = []
        self.vehicles_pos = {_: (None, None) for _ in range(self.n_agents)}
        self._vehicle_dones = [False] * self.n_agents
        self._successful_delivery = 0
        self._total_accepted_orders = 0
        self._total_delivered_orders = 0
        self._total_rejected_orders = 0
        self._vehicle_mileage = [0] * self.n_agents
        self.vehicle_paths, self.vehicle_path_lengths = dijkstra_paths(self._grid_shape[0], self._grid_shape[1])
        self.vehicles_moving = [False for _ in range(self.n_agents)]

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

    def __select_order(self):
        self._total_appeared_orders += 1
        return self.points_locations[np.random.randint(0, self._generated_points)]


    def __init_vehicles(self):
        for vehicle_i in range(self.n_agents):
            self.vehicles_pos[vehicle_i] = self.depot_location

    def __update_vehicles_pos(self, vehicles_actions):
        for vehicle_i, vehicle_action in enumerate(vehicles_actions):
            self.__update_vehicle_pos(vehicle_i, vehicle_action)

    def __update_vehicle_pos(self, vehicle_i, vehicle_action):
        if vehicle_action == 2:  # return to depot: move one step towards depot location
            self.__vehicle_to_depot(vehicle_i)
        elif 3 <= vehicle_action <= self.action_max:  # deliver order i: move one step towards order i
            self.vehicles_moving[vehicle_i] = True
            order_i = vehicle_action - 3
            if self.order_status[order_i] == 0 and self.order_vehicle[order_i] == vehicle_i:
                self.__vehicle_to_order(vehicle_i, order_i)
            else:
                self.rewards[
                    vehicle_i] -= self.invalid_action_penalty  # give large penalty to vehicle for invalid action


    def __update_action_1(self, vehicles_actions):
        # IF MORE THAN ONE ACCEPT ORDERS, assign order to vehicles with minimum cost insertion
        if vehicles_actions.count(1) > 1:
            # penalise if no orders available to be accepted
            if self.order_status.count(0) == 0:
                for vehicle_i, vehicle_action in enumerate(vehicles_actions):
                    if vehicle_action == 1:
                        self.rewards[vehicle_i] -= self.invalid_action_penalty
            # otherwise: assign the open order to cheapest vehicle
            else:
                cheapest_vehicle, new_order_i = self.__cheapest_insertion(
                    vehicles_actions)  # returns the vehicle id and order id of cheapest insertion cost
                for vehicle_i in range(self.n_agents):
                    if vehicle_i == cheapest_vehicle:
                        self.order_status[new_order_i] = 1
                        self.order_vehicle[new_order_i] = cheapest_vehicle
                        self.rewards[vehicle_i] += self.accept_reward
                        self._total_accepted_orders += 1

        # IF ONLY ONE VEHICLE ACCEPT ORDER, assign the order to the only vehicle that accepted the order
        elif vehicles_actions.count(1) == 1:
            # penalise if no orders available to be accepted
            if self.order_status.count(0) == 0:
                for vehicle_i, vehicle_action in enumerate(vehicles_actions):
                    if vehicle_action == 1:
                        self.rewards[vehicle_i] -= self.invalid_action_penalty
            # assign to the correct vehicle
            else:
                for vehicle_i, vehicle_action in enumerate(vehicles_actions):
                    if vehicle_action == 1:  # accept new order
                            for order_i, order_i_status in enumerate(self.order_status):
                                if order_i_status == 0:  # correct order status
                                    self.order_status[order_i] = 1  # assigned to vehicle
                                    self.order_vehicle[order_i] = vehicle_i
                                    self.rewards[vehicle_i] += self.accept_reward
                                    self._total_accepted_orders += 1
                        # update other open order status for other vehicles to 1 (assigned to other vehicles)

        # Otherwise, order expires and becomes inactive again
        else:
            # check if there are any active orders
            if self.order_status.count(0) > 0:
                self._total_rejected_orders += 1
                for vehicle_i in range(self.n_agents):
                    for order_i in range(self.n_orders):
                        if self.order_status[order_i] == 0:
                            self.order_status[order_i] = 4

        # Using minimum cost insertion to decide between vehicles which accept same order (outputs: vehicle_i number)

    def __calculate_TSP_distance(self, route):

        if len(route) <= 1:
            best_route = []
            best_cost = 0
        elif len(route) == 2:
            best_route = route
            a = np.array(route[0])
            b = np.array(route[1])
            best_cost = np.linalg.norm(a - b)
        # elif len(route) == 3:
        #     best_route = [0, 1, 2]
        #     a = np.array(route[0])
        #     b = np.array(route[1])
        #     c = np.array(route[2])
        #     dist_ab = np.linalg.norm(a - b)
        #     dist_bc = np.linalg.norm(b - c)
        #     best_cost = dist_ab + dist_bc
        else:
            best_route, best_cost = sim_annel_solve(np.array(route), 100, 0.9, 0.01, 1) #use sim annel with 87 iterations

        return best_route, best_cost

    def __calculate_insertion_cost(self, route, node):

        if len(route) <= 0:
            updated_route = []
            min_cost = 0
            min_index = 0
        # elif len(route) == 2:
        #     best_route = route
        #     a = np.array(route[0])
        #     b = np.array(route[1])
        #     best_cost = np.linalg.norm(a - b)
        # elif len(route) == 3:
        #     best_route = [0, 1, 2]
        #     a = np.array(route[0])
        #     b = np.array(route[1])
        #     c = np.array(route[2])
        #     dist_ab = np.linalg.norm(a - b)
        #     dist_bc = np.linalg.norm(b - c)
        #     best_cost = dist_ab + dist_bc
        else:
            updated_route, min_cost, min_index = cheapest_insertion(route,node) #use sim annel with 87 iterations

        return updated_route, min_cost, min_index

    def __cheapest_insertion(self, vehicles_action):
        new_order_id = []
        vehicles_insertion_cost = {}

        for vehicle_i, action in enumerate(vehicles_action):
            # if vehicle is still in depot, then all assigned orders are included in route calculation
            if action == 1 and self.vehicles_pos[vehicle_i] == self.depot_location:
                old_route = [self.depot_location]
                new_order = []
                for order_i in range(self.n_orders):
                    if self.order_status[order_i] == 1 and self.order_vehicle[order_i] == vehicle_i:
                        old_route += [self.orders_pos[order_i]]
                    if self.order_status[order_i] == 0:
                        new_order = self.orders_pos[order_i]
                        new_order_id = order_i
                route, old_route_distance = self.__calculate_TSP_distance(old_route)
                _, insertion_cost, _ = self.__calculate_insertion_cost(np.array(route), np.array(new_order))
                vehicles_insertion_cost[vehicle_i] = insertion_cost

            # if vehicle is on journey, then only orders_status = 2 are included in route calculation
            elif action == 1 and self.vehicles_pos[vehicle_i] != self.depot_location:
                old_route = [self.depot_location]
                new_order = []
                for order_i in range(self.n_orders):
                    if self.order_status[order_i] == 1 and self.order_vehicle[order_i] == vehicle_i:
                        old_route += [self.orders_pos[order_i]]
                    if self.order_status[order_i] == 0:
                        new_order = self.orders_pos[order_i]
                        new_order_id = order_i
                route, old_route_distance = self.__calculate_TSP_distance(old_route)
                _, insertion_cost, _ = self.__calculate_insertion_cost(np.array(route), np.array(new_order))
                vehicles_insertion_cost[vehicle_i] = insertion_cost

        assigned_vehicle_id = min(vehicles_insertion_cost, key=vehicles_insertion_cost.get)

        return assigned_vehicle_id, new_order_id

    def __vehicle_to_order(self, vehicle_i, order_i):
        current_pos = self.vehicles_pos[vehicle_i]
        order_i_pos = self.orders_pos[order_i]
        if current_pos == order_i_pos:
            self.order_status[order_i] = 3 #delivered
            self.vehicles_moving[vehicle_i] = False
            self.order_delivered[order_i] = vehicle_i
            self.rewards[vehicle_i] += self.delivery_reward

        # else: #if agent go through another order
        #     for i, order in enumerate(self.orders_pos):
        #         if current_pos == order:
        #             self.order_status[i] = 1
        #             self.rewards[
        #                 vehicle_i] += self.delivery_reward



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
                if j != 2:
                    fill_cell_im(img, self.icon_pkg, self.orders_pos[idx], cell_size=CELL_SIZE)
                else:
                    fill_cell_im(img, self.icon_delivered, self.orders_pos[idx], cell_size=CELL_SIZE)
        # img.show()

        self.images.append(img)
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
        


