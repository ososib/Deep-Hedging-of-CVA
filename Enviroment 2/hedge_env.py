import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EventCallback

from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.env_checker import check_env
from abc import ABC, abstractmethod
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.policies import obs_as_tensor
import scipy.stats as stats
import math
import torch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from stable_baselines3.common.logger import Logger

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None


from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

if TYPE_CHECKING:
    from stable_baselines3.common import base_class

from market_sim import *
from esvar import *

class HedgingEnv(gym.Env):

    def __init__(self, price_model, num_steps=2500, trading_cost_para=0.01,
                 L=100, reward_function=1, T=10, dt=0, a=[], beta_i=0):
        self.action_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=float)#np.float32)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(41,), dtype=float)#np.float32)
        self.price_model = price_model
        self.T=T
        self.t=0
        self.dt=dt
        self.a=a
        self.beta_i=beta_i

        u = np.unique(a)
        self.length=len(u)
        self.K=self.price_model.get_K(self.T, self.t, u, self.beta_i)

        S=self.price_model.get_S(self.T, self.t, u, self.beta_i)
        self.swaps = self.price_model.compute_swaps(self.T,self.t,u,self.K,S,self.length)
        self.Qs = price_model.get_Q(u,self.length,self.t)

        self.n = 0
        self.done = False
        self.num_steps = num_steps
        self.reward_function = reward_function
        hedges_length = 20
        self.h = np.zeros(hedges_length, dtype=float)
        self.trading_cost_para = trading_cost_para
        # 1:a default, 2:e Swaptions, 3:e product
        self.oldQs=self.Qs
        self.oldSwaps=self.swaps
        self.oldCva=np.dot(self.Qs,self.swaps)
        self.dailypnl=np.array([])
        

    def _compute_reward(self, h, Qs, swaps, cva ,nh): #h is currently held hedge, p is current price, op is old price, nh is next hedge. #still need to add variance stuff(?)
        if self.reward_function == 1:
          reward = self.reward1(h, Qs, swaps, cva)
        elif self.reward_function == 2:
          reward = self.reward2(h, Qs, swaps, cva)
        elif self.reward_function == 3:
          reward = self.reward3(h, Qs, swaps, cva)
        elif self.reward_function == 4:
          reward = self.reward4(h, Qs, swaps, cva)
        elif self.reward_function == 5:
          reward = self.reward5(h, Qs, swaps, cva)
        elif self.reward_function == 6:
          reward = self.reward6(h, Qs, swaps, cva)
        elif self.reward_function == 7:
          reward = self.reward7(h, Qs, swaps, cva)
        else:
          reward = 0
        tradingcost = np.dot(np.abs(nh-h),np.concatenate((self.oldQs,self.oldSwaps)))*self.trading_cost_para
        reward = reward - tradingcost
        return reward
        """
        elif self.reward_function == 5:
          reward = self.reward5(h, Qs, swaps, cva)
        elif self.reward_function == 6:
          reward = self.reward6(h, Qs, swaps, cva)
        elif self.reward_function == 7:
          reward = self.reward7(h, Qs, swaps, cva)
        elif self.reward_function == 8:
          reward = self.reward8(h, Qs, swaps, cva)
        elif self.reward_function == 9:
          reward = self.reward9(h, Qs, swaps, cva)
        elif self.reward_function == 10:
          rwward = self.reward10(h, Qs, swaps, cva)
        """

    def comp_pnl(self,h, Qs, swaps, cva):
      # Pnl(t) =  (P(t) - P(t-1)) - ( alpha(t-1) * B(t) + beta(t-1) * A(t) ) - ( alpha(t-1)*B(t-1) + beta(t-1) * A(t-1) )
      #faktorn 2 kommer från omskalning av action space (borrtaget för now (2024-04-26))
      return (cva-self.oldCva)-np.dot(h,np.concatenate((Qs,swaps))-np.concatenate((self.oldQs,self.oldSwaps)))

    def reward1(self, h, Qs, swaps, cva):
      pnl = self.comp_pnl(h, Qs, swaps, cva)
      return -np.abs(pnl)

    def reward2(self, h, Qs, swaps, cva):
      pnl = self.comp_pnl(h, Qs, swaps, cva)
      return pnl

    def reward3(self, h, Qs, swaps, cva):
      pnl = self.comp_pnl(h, Qs, swaps, cva)
      return np.min([pnl,0])

    def reward4(self, h, Qs, swaps, cva):
      pnl = self.comp_pnl(h, Qs, swaps, cva)
      return -(pnl)**2
    
    def reward5(self, h, Qs, swaps, cva):
      a,b = self.price_model.sim_many_prices(0.03, self.oldprice[0], self.oldprice[1])
      sim_contract = a*b
      sim_hedge_err = (sim_contract-self.oldprice[2])-((2*h[0]*a+2*h[1]*b)-(2*h[0]*self.oldprice[0]+2*h[1]*self.oldprice[1]))
      return -np.var(sim_hedge_err)

    def reward6(self, h, Qs, swaps, cva):
      if self.n < 30:
        self.dailypnl = np.append(self.dailypnl,[-self.comp_pnl(h, Qs, swaps, cva)])
        return 0
      else:
        self.dailypnl = np.append(self.dailypnl[1:],[-self.comp_pnl(h, Qs, swaps, cva)])
        #var99_10days=var99 * np.sqrt(10)
        return -calculate_var(self.dailypnl, 99)

    def reward7(self, h, Qs, swaps, cva):
      if self.n < 30:
        self.dailypnl = np.append(self.dailypnl,[-self.comp_pnl(h, Qs, swaps, cva)])
        return 0
      else:
        self.dailypnl = np.append(self.dailypnl[1:],[-self.comp_pnl(h, Qs, swaps, cva)])
        return -calculate_es(self.dailypnl,alpha= 0.01)

    """
    def reward8(self, h, Qs, swaps, cva):
      pnl = self.comp_pnl(h, Qs, swaps, cva)
      return -np.sqrt(np.abs(pnl))

    def reward9(self, h, Qs, swaps, cva):
      pnl = self.comp_pnl(h, Qs, swaps, cva)
      return pnl-(pnl)**2-0.25

    """
    def step(self, delta_h):
        self.n += 1
        self.t += self.dt
        new_h = self.h#actually old hedge xdD
        self.h = delta_h
        self.oldQs=self.Qs
        self.oldSwaps=self.swaps
        self.oldCva=np.dot(self.Qs,self.swaps)
        
        a=self.a
        u = np.unique(a[self.n:])

        self.price_model.compute_next_intensity_and_rate()

        S=self.price_model.get_S(self.T, self.t, u, self.beta_i)
        self.swaps = self.price_model.compute_swaps(self.T,self.t,u,self.K,S,self.length)
        self.Qs = self.price_model.get_Q(u,self.length,self.t)
        cva=np.dot(self.Qs,self.swaps)

        reward = self._compute_reward(self.h, self.Qs, self.swaps, cva, new_h)

        if self.n == self.num_steps:
            self.done = True
        if(min(self.Qs)<0):
          print("ERROR! Negative Q")
        if(min(self.swaps)<0):
          print("ERROR! Negative Swap")


        ##TODO
        state = np.concatenate((self.Qs, self.swaps, np.array([cva]), self.h))
        return state, reward, self.done, False, {}

    def reset(self, seed=None, options=None):
        self.price_model.reset()

        self.t=0
        a=self.a
        u = np.unique(a)
        self.K=self.price_model.get_K(self.T, self.t, u, self.beta_i)
        S=self.price_model.get_S(self.T, self.t, u, self.beta_i)
        self.swaps = self.price_model.compute_swaps(self.T,self.t,u,self.K,S,self.length)
        self.Qs = self.price_model.get_Q(u,self.length,self.t)

        self.n = 0
        self.done = False
        hedges_length = 20
        self.h = np.ones(hedges_length, dtype=float)*0.000000001
        # 1:a default, 2:e Swaptions, 3:e product
        self.oldQs=self.Qs
        self.oldSwaps=self.swaps
        self.oldCva=np.dot(self.Qs,self.swaps)
        self.dailypnl=np.array([])
        state = np.concatenate((self.Qs, self.swaps, np.array([np.dot(self.Qs,self.swaps)]),self.h))
        return state, {}
