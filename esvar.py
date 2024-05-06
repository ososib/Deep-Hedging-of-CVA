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
import hedge_env
import callbacks


def calculate_es(returns, alpha):

    # Sort the returns in ascending order
    #returns_sorted = np.sort(returns)

    # Compute the index corresponding to the quantile
    #Var = np.percentile(returns_sorted, percent)
    #alpha = 0.99 # VaR/ES Level for the loss distribution

      mu, std = stats.norm.fit(returns)
      ES = mu+(std*1/np.sqrt(2*np.pi)*np.exp(-0.5*stats.norm.ppf(alpha)**2))/(1-alpha)

      return ES

def calculate_var(returns, percent):
    """
    Calculate the Value at Risk (VaR) given an array of returns and a confidence level.

    Parameters:
    - returns: numpy array of returns
    - confidence_level: float representing the confidence level (e.g., 0.01 for 99% confidence level)

    Returns:
    - var: Value at Risk (VaR) at the given confidence level
    """
    # Sort the returns in ascending order
    #returns_sorted = np.sort(returns)

    # Compute the index corresponding to the quantile
    #Var = np.percentile(returns_sorted, percent)
    #Look for the 99th percentile for the loss distribution
    Var = np.percentile(returns, percent) #should give the same

    return Var
