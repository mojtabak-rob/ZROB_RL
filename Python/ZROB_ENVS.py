from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numpy import linalg as LA

class ZrobEnv(gym.Env):
    def __init__(self, max_silence=2, min_silence=0.2, midi_seq=[1,0.3,1.7,0.5,0.2,0.3]):

        self.num_notes = len(midi_seq)
        self.max_silence = max_silence
        self.min_silence = min_silence
        self.midi_seq = midi_seq
        

        low = np.zeros((self.num_notes), dtype=np.float32) # type: ignore
        high = np.ones((self.num_notes), dtype=np.float32)*self.max_silence

        self.action_space = spaces.Box(low=0, high=self.max_silence, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, u):
        obs = self.state
        done = True
        cost = 0
        midi_seq = self.midi_seq
        
        for i in range(len(obs)):
            if obs[i]==0:
                obs[i]=u
                cost = (len(obs)-i)*abs(u-midi_seq[i])**2
                if abs(u-midi_seq[i])<0.1:
                    done=False
                break

        #for i in range(len(obs)):
            #cost += (obs[i]-midi_seq[i])**2

        self.state = obs
        return self._get_obs(), -cost, done, False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        new = np.zeros((self.num_notes), dtype=np.float32)
        self.state = new

        return self._get_obs()

    def _get_obs(self):

        return self.state


class ZrobEnv2(gym.Env):
    def __init__(self,DoF=2):

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
        dis = obs-self.target
        cost1=(LA.norm(dis)**2-1)**2
        #cost2=0
        #if LA.norm(dis)>0.01:
            #cost2=(u[0]/LA.norm(u)+dis[1]/LA.norm(dis))**2 + (u[1]/LA.norm(u)-dis[0]/LA.norm(dis))**2
        #cost3=(LA.norm(u)**2-0.09*LA.norm(dis)**2)**2
        #cost=cost1+5*cost2+cost3
        cost=cost1
        done=False
        
        return self._get_obs(), -cost, done, False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        new = np.random.uniform(low=0, high=10, size=self.DoF)
        self.state = new

        return self._get_obs()

    def _get_obs(self):
        return self.state