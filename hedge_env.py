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

import market_sim
import esvar

class HedgingEnv(gym.Env):

    def __init__(self, price_model, num_steps=2520, trading_cost_para=0.01,
                 L=100, reward_function=1):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(5,), dtype=float)
        self.price_model = price_model
        self.current_prices = price_model.get_current_price()
        self.n = 0
        self.done = False
        self.num_steps = num_steps
        self.reward_function = reward_function
        self.h = np.array([0.0, 0.0])
        self.trading_cost_para = trading_cost_para
        self.oldprice = [self.current_prices[0], self.current_prices[1], self.current_prices[0]*self.current_prices[1]]
        self.dailypnl=np.array([])

    def _compute_reward(self, h, p ,nh): #h is currently held hedge, p is current price, op is old price, nh is next hedge. #still need to add variance stuff(?)
        match self.reward_function:
          case 1: reward = self.reward1(h,p)
          case 2: reward = self.reward2(h,p)
          case 3: reward = self.reward3(h,p)
          case 4: reward = self.reward4(h,p)
          case 5: reward = self.reward5(h,p)
          case 6: reward = self.reward6(h,p)
          case 7: reward = self.reward7(h,p)
          case 8: reward = self.reward8(h,p)
          case 9: reward = self.reward9(h,p)
          case 10: reward = self.reward10(h,p)
          case 11: reward = self.reward11(h,p)
          case 12: reward = self.reward12(h,p)
          case 13: reward = self.reward13(h,p)
          case 14: reward = self.reward14(h,p)
          case 15: reward = self.reward15(h,p)
          case 16: reward = self.reward16(h,p)
          case 17: reward = self.reward17(h,p)
          case 18: reward = self.reward18(h,p)
          case 19: reward = self.reward19(h,p)
          case 20: reward = self.reward20(h,p)
          case 21: reward = self.reward21(h,p)
          case 21: reward = self.reward22(h,p)
          case _: reward = 0
        #tradingcost = np.abs(np.abs(2*nh[0]-2*h[0])*p[0]+np.abs(2*nh[1]-2*h[1])*p[1])*self.trading_cost_para
        #sellvalue =  p[2]+(h[0]*p[0]+h[1]*p[1])
        #reward = sellvalue-self.buyvalue
        #reward = -100*(sellvalue-self.buyvalue)**2
        #if self.n==1:
        #  reward = -100*(p[2]-(h[0]*p[0]+h[1]*p[1]))**2
        #self.buyvalue =  p[2]+(nh[0]*p[0]+nh[1]*p[1])
        #self.price = np.append(self.price,[p[2]])
        #reward = np.min([pminush,0]) - tradingcost  # same as above, but does not punish under-hedging, as that gives profit
        tradingcost = np.abs(np.abs(2*nh[0]-2*h[0])*self.oldprice[0]+np.abs(2*nh[1]-2*h[1])*self.oldprice[1])*self.trading_cost_para #i feel ok with abs for tc.
        reward = reward - tradingcost
        return reward

    def comp_pnl(self,h,p):
      # Pnl(t) =  (P(t) - P(t-1)) - ( alpha(t-1) * B(t) + beta(t-1) * A(t) ) - ( alpha(t-1)*B(t-1) + beta(t-1) * A(t-1) )
      #faktorn 2 kommer från omskalning av action space
      return (p[2]-self.oldprice[2])-((2*h[0]*p[0]+2*h[1]*p[1])-(2*h[0]*self.oldprice[0]+2*h[1]*self.oldprice[1]))

    def reward1(self, h, p):
      pnl = self.comp_pnl(h,p)
      return -np.abs(pnl)

    def reward2(self, h, p):
      pnl = self.comp_pnl(h,p)
      return pnl

    def reward3(self, h, p):
      pnl = self.comp_pnl(h,p)
      return np.min([pnl,0])

    def reward4(self, h, p):
      pnl = self.comp_pnl(h,p)
      return -(pnl)**2

    def reward5(self, h, p):
      a,b = self.price_model.sim_many_prices(0.03, self.oldprice[0], self.oldprice[1])
      sim_contract = a*b
      sim_hedge_err = (sim_contract-self.oldprice[2])-((2*h[0]*a+2*h[1]*b)-(2*h[0]*self.oldprice[0]+2*h[1]*self.oldprice[1]))
      return -np.var(sim_hedge_err)

    def reward6(self, h, p):
      if self.n < 30:
        self.dailypnl = np.append(self.dailypnl,[self.comp_pnl(h,p)])
        return 0
      else:
        self.dailypnl = np.append(self.dailypnl[1:],[self.comp_pnl(h,p)])
        #var99_10days=var99 * np.sqrt(10)
        return calculate_var(self.dailypnl, 1)

    def reward7(self, h, p):
      if self.n < 30:
        self.dailypnl = np.append(self.dailypnl,[self.comp_pnl(h,p)])
        return 0
      else:
        self.dailypnl = np.append(self.dailypnl[1:],[self.comp_pnl(h,p)])
        return calculate_es(self.dailypnl,alpha= 0.01)

    def reward8(self, h, p):
      pnl = self.comp_pnl(h,p)
      return -np.sqrt(np.abs(pnl))

    def reward9(self, h, p):
      pnl = self.comp_pnl(h,p)
      return pnl-(pnl)**2-0.25

    def reward10(self, h, p):
      a,b = self.price_model.sim_many_prices(0.03, self.oldprice[0], self.oldprice[1])
      sim_contract = a*b
      sim_hedge_err = (sim_contract-self.oldprice[2])-((2*h[0]*a+2*h[1]*b)-(2*h[0]*self.oldprice[0]+2*h[1]*self.oldprice[1]))
      return np.mean(sim_hedge_err)

    def reward11(self, h, p):
      a,b = self.price_model.sim_many_prices(0.03, self.oldprice[0], self.oldprice[1])
      sim_contract = a*b
      sim_hedge_err = (sim_contract-self.oldprice[2])-((2*h[0]*a+2*h[1]*b)-(2*h[0]*self.oldprice[0]+2*h[1]*self.oldprice[1]))
      return -np.abs(np.mean(sim_hedge_err))

    def reward12(self, h, p):
      a,b = self.price_model.sim_many_prices(0.03, self.oldprice[0], self.oldprice[1])
      sim_contract = a*b
      sim_hedge_err = (sim_contract-self.oldprice[2])-((2*h[0]*a+2*h[1]*b)-(2*h[0]*self.oldprice[0]+2*h[1]*self.oldprice[1]))
      return -np.std(sim_hedge_err)

    def reward13(self, h, p):
      a,b = self.price_model.sim_many_prices(0.03, self.oldprice[0], self.oldprice[1])
      sim_contract = a*b
      sim_hedge_err = (sim_contract-self.oldprice[2])-((2*h[0]*a+2*h[1]*b)-(2*h[0]*self.oldprice[0]+2*h[1]*self.oldprice[1]))
      return -np.mean(sim_hedge_err)**2-np.var(sim_hedge_err)

    def reward14(self, h, p):
      a,b = self.price_model.sim_many_prices(0.03, self.oldprice[0], self.oldprice[1])
      sim_contract = a*b
      sim_hedge_err = (sim_contract-self.oldprice[2])-((2*h[0]*a+2*h[1]*b)-(2*h[0]*self.oldprice[0]+2*h[1]*self.oldprice[1]))
      return -np.abs(np.mean(sim_hedge_err))-np.std(sim_hedge_err)

    def reward17(self, h, p):
      abserror = (p[2]-(2*h[0]*p[0]+2*h[1]*p[1]))
      return -np.abs(abserror)

    def reward18(self, h, p):
      abserror = (p[2]-(2*h[0]*p[0]+2*h[1]*p[1]-self.oldprice[2]))
      return -np.abs(abserror)

    def reward17(self, h, p):
      abserror = (p[2]-(2*h[0]*p[0]+2*h[1]*p[1]))
      return -(abserror)**2

    def reward18(self, h, p):
      abserror = (p[2]-(2*h[0]*p[0]+2*h[1]*p[1]-self.oldprice[2]))
      return -(abserror)**2

    def reward19(self, h, p):
      abserror = (2*p[2]-(2*h[0]*p[0]+2*h[1]*p[1]))
      return -(abserror)**2

    def reward20(self, h, p):
      abserror = (2*p[2]-(2*h[0]*p[0]+2*h[1]*p[1]))
      return abserror-(abserror)**2-0.25

    def reward21(self, h, p):
      a,b = self.price_model.sim_many_prices(0.03, self.oldprice[0], self.oldprice[1])
      sim_contract = a*b
      sim_abserror = (2*sim_contract-(2*h[0]*a+2*h[1]*b))
      return -np.mean(sim_abserror)**2

    def reward22(self, h, p):
      a,b = self.price_model.sim_many_prices(0.03, self.oldprice[0], self.oldprice[1])
      sim_contract = a*b
      sim_hedge_err = (sim_contract-self.oldprice[2])-((2*h[0]*a+2*h[1]*b)-(2*h[0]*self.oldprice[0]+2*h[1]*self.oldprice[1]))
      return calculate_var(sim_hedge_err, 1)

    def step(self, delta_h):
        new_h = self.h#actually old hedge xdD
        self.h = delta_h
        self.oldprice = [self.current_prices[0], self.current_prices[1], self.current_prices[0]*self.current_prices[1]]
        self.price_model.compute_next_price(0.03) #gotta figure this out
        self.current_prices = self.price_model.get_current_price()
        current_price_contract = self.current_prices[0] * self.current_prices[1]
        p = [self.current_prices[0], self.current_prices[1], self.current_prices[0]*self.current_prices[1]]
        self.n += 1
        if self.n == self.num_steps:
            self.done = True
        reward = self._compute_reward(self.h,p, new_h)
        state = np.array([self.current_prices[0], self.current_prices[1], current_price_contract, 2*self.h[0], 2*self.h[1]])
        if self.current_prices[1]>=1:
          self.done = True
        return state, reward, self.done, False, {}

    def reset(self, seed=None, options=None):
        #self.price_model.reset()
        self.price_model.new(np.random.normal(0.15, 0.05),np.random.normal(0.5, 0.1)) # this is scuffed
        self.n = 0
        self.done = False
        self.dailypnl=np.array([])
        self.current_prices = self.price_model.get_current_price()
        self.h = np.array([0.0, 0.0])
        current_price_contract = self.current_prices[0] * self.current_prices[1]
        self.oldprice = [self.current_prices[0], self.current_prices[1], self.current_prices[0]*self.current_prices[1]]
        state = np.array([self.current_prices[0], self.current_prices[1], current_price_contract, 0.0000001, 0.0000001])
        return state, {}