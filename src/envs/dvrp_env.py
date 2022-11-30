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
        raise NotImplementedError

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
        raise NotImplementedError

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
        


