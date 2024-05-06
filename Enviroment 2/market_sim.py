"""## Define market simulator"""

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


"""## Define market simulator"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import ncx2
import scipy.stats as stats
from scipy.stats import norm

class GenericAssetPriceModel(ABC):
    @abstractmethod
    def get_current_price(self):
        pass

    @abstractmethod
    def compute_next_price(self, *action):
        pass

    @abstractmethod
    def reset(self):
        pass

class BOTH(GenericAssetPriceModel):
    def __init__(self, rho, dt, lambda0, kappa, mu, nu, jump_alpha, jump_gamma, r0, sigma, n=2520):
      self.n = n
      self.dt = dt
      self.rho = rho
      self.lambda0 = lambda0
      self.kappa = kappa
      self.mu = mu
      self.nu = nu
      self.jump_alpha = jump_alpha
      self.jump_gamma = jump_gamma
      self.current_lambda = lambda0
      u=np.random.uniform(0,1,n)#int(round(n/8)))
      t=np.cumsum(-1/jump_alpha*np.log(u))
      B=np.random.exponential(jump_gamma,n)
      self.j=np.zeros(n)
      tindex=1
      while int(np.round(t[tindex]))<n:
        self.j[int(np.round(t[tindex]))]=np.sum(B[int(np.round(t[tindex-1])):int(np.round(t[tindex]))])#np.sum(B[0:int(np.round(t[tindex]))])-np.sum(B[0:int(np.round(t[tindex-1]))])
        tindex = tindex+1
      self.t=0
      self.r_0=r0
      self.current_r=r0
      self.sigma = sigma



    def compute_next_intensity_and_rate(self):
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
        x3 = self.rho*x1+np.sqrt(1-self.rho**2)*x2
        """
        upsilon = 4 * self.mu * self.kappa / self.nu**2
        non_central = 4 * self.kappa * np.exp(-self.kappa * self.dt) / (self.nu**2 * (1 - np.exp(-self.kappa * self.dt))) * self.current_lambda
        var = ncx2.rvs(upsilon, non_central)
        self.current_lambda = self.nu**2 * (1-np.exp(-self.kappa * self.dt)) / (4 * self.kappa) * var + self.j[self.t]
        """
        #https://quant.stackexchange.com/questions/8114/monte-carlo-simulating-cox-ingersoll-ross-process
        cee = 1 + self.kappa*self.dt+self.nu*np.sqrt(self.dt)*np.abs(x1)/np.sqrt(self.current_lambda)
        self.current_lambda = self.current_lambda + self.kappa*(self.mu-self.current_lambda)*self.dt/cee + self.nu*np.sqrt(self.current_lambda)*np.sqrt(self.dt)*x1/cee + self.j[self.t]
        self.t = self.t+1
        self.current_r = self.current_r*np.exp(self.sigma*np.sqrt(self.dt)*x3)#self.current_r+self.current_r*sigma*np.sqrt(dt)*x3

    def alphabeta(self,h,T,t):
      denom=(2*h+(self.kappa+h)*(np.exp(h*(T-t))-1))
      beta = 2*(np.exp(h*(T-t))-1)/denom
      A = (2*h*np.exp((self.kappa+h)*(T-t)/2)/denom)**(2*self.kappa*self.mu/(self.nu**2))
      alpha = A*(2*h*np.exp((self.kappa+h+2*self.jump_gamma)*(T-t)/2)/(2*h+(self.kappa+h+2*self.jump_gamma)*(np.exp(h*(T-t))-1)))**(2*self.jump_alpha*self.jump_gamma/(self.nu**2+2*self.kappa*self.jump_gamma-2*self.jump_gamma**2))
      return alpha,beta


    def compute_next_price(self,T,t):
        h = np.sqrt(self.kappa**2+2*self.nu**2)
        alpha, beta = self.alphabeta(h,T,t)
        return np.exp(-beta*self.current_lambda)*alpha


    def reset(self):
        self.current_lambda = self.lambda0
        n = self.n
        u=np.random.uniform(0,1,int(round(n/8)))
        t=np.cumsum(-1/self.jump_alpha*np.log(u))
        B=np.random.exponential(self.jump_gamma,n)
        self.j=np.zeros(n)
        tindex=1
        while int(np.round(t[tindex]))<n:
          self.j[int(np.round(t[tindex]))]=np.sum(B[0:int(np.round(t[tindex]))])-np.sum(B[0:int(np.round(t[tindex-1]))])
          tindex = tindex+1
        self.t=0
        self.current_r=self.r_0

    def get_current_rate(self):
        return self.current_r

    def get_current_price(self):
        pass

    def get_K(self,T,t,u,beta_i):
      r = self.current_r
      den=0
      for j in range(1,len(u)):
        den=den+beta_i*np.exp(-r*(u[j]-t))
      K=(1-np.exp(-r*(T-t)))/den
      return K

    def get_S(self,T,t,u,beta_i):
      r = self.current_r
      S=np.zeros(len(u))
      for k in range(0,len(u)-1):
        den=0
        for j in range(k+1,len(u)):
          den=den+beta_i*np.exp(-r*(u[j]-t))
        s=(np.exp(-r*(u[k]-t))-np.exp(-r*(T-t)))/den
        S[k+1]=s
      return S


    def get_Q(self,u,length,t):
      Q=np.zeros(length-1)
      adj=length-len(u)
      for j in range(1,len(u)):
        if(t>u[j]):
          Q[adj+j-1]=0
        elif(t<u[j-1]):
          Q[adj+j-1]=(self.compute_next_price(u[j-1],t) - self.compute_next_price(u[j],t))
        else:
          Q[adj+j-1]=(1 - self.compute_next_price(u[j],t))
      return Q

    def compute_swaps(self,T,t,u,K,S,length):
      sigma = self.sigma
      r = self.current_r

      #for i in range (2520):
       # t=t+dt
       # a.append(np.floor(t/(beta_i))*(beta_i))
        #a=[x-a[0] for x in a]
        #length=len(np.unique(a))

      
      swaps=np.zeros(length-1)
      adj=length-len(u)
      for k in range(1,len(u)):
        swaps[adj+k-1]=self.european_swaption_price(T,t,u[k],u[k:],sigma,r,K,S[k])
      return swaps


    def european_swaption_price(self,final_time, current_time, maturity_time, tenor, volatility, short_rate, strike_rate, swap_rate):
        d1 = (np.log(swap_rate / strike_rate) + 0.5 * volatility**2 * (maturity_time-current_time)) / (volatility * np.sqrt(maturity_time-current_time))
        d2 = d1 - volatility * np.sqrt(maturity_time-current_time)
        discount = self.discount_curve(tenor, current_time, short_rate)
        pv = discount  * (swap_rate * norm.cdf(d1) - strike_rate * norm.cdf(d2))
        return pv

    # Sample discount curve function (for simplicity, using a flat curve here)
    def discount_curve(self, tenor, current_time, short_rate):
        discount_terms= [np.exp(-short_rate*(T-current_time)) for T in tenor]
        return  sum(discount_terms)# flat discount rate of 5%
