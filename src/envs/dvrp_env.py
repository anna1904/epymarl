# from .multiagentenv import MultiAgentEnv
from gym import Env, spaces
import json
import random
import os
import numpy as np


class DVRPEnv(Env):
    def __init__(self):
        self.observation_space = [spaces.Box(0, 3, shape=(2,), dtype=int)]
        self.action_space = [spaces.Discrete(4)]
        self.n_agents = 2
        self.episode_limit = 200

    def step(self, actions):
        return 1, {}

    def get_stats(self):
        return {}

    def get_obs(self):
        return []

    def get_obs_agent(self, agent_id):
        # print(self.observations[agent_id].flatten().shape)
        # print(self.get_obs_size())
        return self.observations[agent_id].flatten()

    def get_obs_size(self):
        return int(np.prod(next(iter(self.env.observation_spaces.values())).shape))

    def get_state(self):
        return np.concatenate([self.get_obs_agent(o) for o in self.env.agents],axis=0)

    def get_state_size(self):
        return  self.get_obs_size()*self.env.num_agents

    def get_avail_actions(self):
        return [[1]*self.get_total_actions()]*self.n_agents

    def get_avail_agent_actions(self, agent_id):
        return [1]*self.get_total_actions()

    def get_total_actions(self):
        return self.num_discrete_acts

    def reset(self):
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        pass

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
        


