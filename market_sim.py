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
    def __init__(self, r_0, dt, sigma_H, alpha, s_0, r, sigma, mu_J, lambda_J, sigma_J, rho):
        self.current_r = r_0
        self.r_0 = r_0
        self.dt = dt
        self.sigma_H = sigma_H
        self.alpha = alpha
        self.s_0 = s_0
        self.r = r
        self.sigma = sigma
        self.mu_J = mu_J
        self.lambda_J = lambda_J
        self.sigma_J = sigma_J
        self.current_price = s_0
        self.rho = rho


    def compute_next_price(self, theta):
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
        x3 = self.rho*x1+np.sqrt(1-self.rho**2)*x2
        i = np.sqrt(self.dt)*x1
        i2 = np.sqrt(self.dt)*x3
        j = np.random.poisson(self.dt*self.lambda_J)
        drift = self.r - self.lambda_J*(self.mu_J + self.sigma_J**2/2) - 0.5*self.sigma**2;
        jump = j*np.random.normal(self.mu_J,self.sigma_J)#j*np.random.normal(self.mu_J - self.sigma_J**2/2,self.sigma_J)
        new_price = self.current_price * np.exp(drift * self.dt + self.sigma * i + jump)
        self.current_price = new_price
        self.current_r = self.current_r + (theta-self.alpha*self.current_r)*self.dt+self.sigma_H*i2#np.random.normal(0, np.sqrt(self.dt))

    def sim_many_prices(self, theta, currentr, currentprice):
        x1 = np.random.normal(0, 1, (10000,1))
        x2 = np.random.normal(0, 1, (10000,1))
        x3 = self.rho*x1+np.sqrt(1-self.rho**2)*x2
        i = np.sqrt(self.dt)*x1
        i2 = np.sqrt(self.dt)*x3
        j = np.random.poisson(self.dt*self.lambda_J, (10000,1))
        drift = self.r - self.lambda_J*(self.mu_J + self.sigma_J**2/2) - 0.5*self.sigma**2;
        jump = j*np.random.normal(self.mu_J,self.sigma_J)#j*np.random.normal(self.mu_J - self.sigma_J**2/2,self.sigma_J)
        new_prices = currentprice * np.exp(drift * self.dt + self.sigma * i + jump)
        new_rs = currentr + (theta-self.alpha*currentr)*self.dt+self.sigma_H*i2#np.random.normal(0, np.sqrt(self.dt))
        return new_rs, new_prices

    def reset(self):
        self.current_r = self.r_0
        self.current_price = self.s_0

    def get_current_price(self):
        return self.current_r, self.current_price

    def new(self, r0, s0):
        self.current_r = r0
        self.current_price = s0