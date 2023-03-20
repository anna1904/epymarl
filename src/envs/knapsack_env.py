# from .multiagentenv import MultiAgentEnv
from gym import Env, spaces
import numpy as np
from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
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

class BinaryKnapsackEnv(Env):
    def __init__(self, n_agents = 2, N = 5):

        np.random.seed(1) #random seed

        self.n_agents = n_agents
        self.N = N
        self.item_weights = np.random.randint(1, 25, size=self.N)
        self.item_values = np.random.randint(0, 25, size=self.N)
        self.max_weight = 50
        self.current_weight = [0 for _ in range(n_agents)]
        self._max_reward = 10000
        self.mask = True
        self.seed = 0
        self.step_count = 0
        self.item_numbers = np.arange(self.N)
        self.randomize_params_on_reset = False
        self.item_limits = [np.ones(self.N, dtype=np.int32) for _ in range(self.n_agents)]

        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(0, self.max_weight, shape=(3, self.N + 1), dtype=np.int32) for _ in range(self.n_agents)])

        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.N) for _ in range(self.n_agents)])

    def get_avail_agent_actions(self, agent_id):
        mask = np.where(self.current_weight[agent_id] + self.item_weights > self.max_weight, 0, 1).astype(np.uint8)
        mask = np.where(self.item_limits[agent_id] > 0, mask, 0)
        return mask

    def step(self, items):
        vehicles_action = list(items)
        self.step_count += 1
        rewards = [0] * self.n_agents
        dones = [False] * self.n_agents

        for agent, item in enumerate(items):
            # Check item limit
            self.current_weight[agent] += self.item_weights[item]
            rewards[agent] = self.item_values[item]
            if self.current_weight[agent] == self.max_weight:
                dones[agent] = True

        self._update_state(items)

        for agent in range(self.n_agents):
            mask = self.get_avail_agent_actions(agent)
            if np.count_nonzero(mask) == 0:
                dones[agent] = True

        return self.state, rewards, dones, {}

    def _update_state(self, items):

        _obs = []
        for i, item in enumerate(items):
            self.item_limits[i][item] -= 1
            state_items = np.vstack([
                self.item_weights,
                self.item_values,
                self.item_limits[i]
            ], dtype=np.int32)

            state = np.hstack([
                state_items,
                np.array([[self.max_weight],
                            [self.current_weight[i]],
                            [0]  # Serves as place holder
                            ])
            ], dtype=np.int32)
            _obs.append(state)

        self.state = _obs.copy()
        return self.state

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
        for a in range(self.n_agents):
            state_items = np.vstack([
                self.item_weights,
                self.item_values,
                self.item_limits[a]
            ], dtype=np.int32)

            state = np.hstack([
                state_items,
                np.array([[self.max_weight],
                          [self.current_weight[a]],
                          [0]  # Serves as place holder
                          ])
            ], dtype=np.int32)
            _obs.append(state)

        self.state = _obs.copy()
        return _obs

    def get_state_size(self):
        return self.get_obs_size()*self.n_agents

    def get_total_actions(self):
        return self.action_max

    def reset(self):

        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N)
            self.item_values = np.random.randint(0, 100, size=self.N)
        self.current_weight = [0] * self.n_agents
        self.item_limits = [np.ones(self.N, dtype=np.int32) for _ in range(self.n_agents)]
        self.step_count = 0
        self.get_state()
        return self.state

    def render(self, mode = 'human', close = False):

        # total_value = 0
        # total_weight = 0
        # for i in range(self.N):
        #     if i in self._collected_items:
        #         total_value += self.item_values[i]
        #         total_weight += self.item_weights[i]
        # print(self._collected_items, total_value, total_weight)
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
        


