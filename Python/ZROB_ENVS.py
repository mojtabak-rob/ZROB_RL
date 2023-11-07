from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numpy import linalg as LA

class ZrobEnv(gym.Env):
    def __init__(self,DoF=4):

        self.DoF=DoF
        
        self.target=np.ones((DoF),dtype=np.float32)*5

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.DoF,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=10, shape=(self.DoF,), dtype=np.float32)

    def step(self, u):
        obs = self.state
        
        lower_bound = self.observation_space.low # type: ignore
        upper_bound = self.observation_space.high # type: ignore
        new = obs+u
        new = np.clip(new, lower_bound, upper_bound)
        self.state=new
        cost=0
        
        done=False
        
        return self._get_obs(), -cost, done, False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        new = np.random.uniform(low=0, high=10, size=self.DoF)
        self.state = new

        return self._get_obs()

    def _get_obs(self):
        return self.state