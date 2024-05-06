import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch as th

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
from hedge_env import *
from callbacks import *
from esvar import *

import pandas as pd
"""## Define Environment

Qs, 0:9
swaps 10:19
cva 20
Q_quantity 21:30
Swaps_quantity 31:
"""


dt = 1/252 # time step
lambda0=0.035
kappa=0.35
mu=0.045
nu=0.15
jump_alpha=0.001 #his values here are crazy borgir
jump_gamma=0.0005

T=10

t=0

beta_i=T/(10)

rho = 0

r0=0.03
sigma=0.1

a=[]
for i in range (2520):
  t=t+dt
  a.append(np.floor(t/(beta_i))*(beta_i))
#a=[x-a[0] for x in a]
length=len(np.unique(a))
#print(length)
t=0



for rho in [-1, -0.75,-0.5,0.25,0,0.25,0.5,0.75,1]:
    
    df = pd.DataFrame({},
                      index=['Variance Total Cost', 'Variance Total Cost Without First', 'Mean Square Total Cost', 'Mean Total Cost','Value at Risk', 'Value at Risk Without First', 'Expected Shortfall', 'Expected Shortfall Without First', 'Mean Turnover'])
    
    
    trading_cost = 0
    
    
    apm = BOTH(dt = dt, rho=rho, lambda0=lambda0, kappa=kappa, mu=mu, nu=nu, jump_alpha=jump_alpha, jump_gamma=jump_gamma, r0=r0, sigma=sigma)
    
    ####################################################################################################################################################
    ####################################################################################################################################################
    
    for treward in [7]:
            
        env = HedgingEnv(apm, trading_cost_para=trading_cost, reward_function=treward,T=T, dt=dt, a=a, beta_i=beta_i)
        eval_Env = HedgingEnv(apm, trading_cost_para=trading_cost, reward_function=treward, T=T, dt=dt, a=a, beta_i=beta_i)
        
        # If the environment don't follow the interface, an error will be thrown
        check_env(env, warn=True)
        
        
        # Create log dir
        log_dir_0 = "/local/home/osmanoscar/CVA_results_04_29/tmp/"+'rho'+str(rho)+'/'+ 'rwf' +str(treward)+ "/0/" # change wehn script to depend on reward function, rho, and tc
        os.makedirs(log_dir_0, exist_ok=True) #change exists?
        
        log_dir_1 = "/local/home/osmanoscar/CVA_results_04_29/tmp/"+'rho'+str(rho)+'/'+ 'rwf' +str(treward)+"/1/" # change wehn script to depend on reward function, rho, and tc
        os.makedirs(log_dir_1, exist_ok=True) #change exists?
        
    
        
        # Wrap the environment
        # Logs will be saved in log_dir/monitor.csv
        env = Monitor(env, log_dir_0)#, log_dir)
        eval_Env = Monitor(eval_Env, log_dir_1) #?
        #eval_Env_Bench = Monitor(eval_Env_Bench, log_dir_2) #?
        
        """## Train agent"""
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[512,512], vf=[512,512]))
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,  verbose=0)
        #from stable_baselines3 import SAC
        #model = SAC("MlpPolicy", env, verbose=0) #performs badly. I think because it needs more hyperparameter tuning than PPO.
        
        #callback = EvalCallback(eval_env = eval_Env, eval_freq = 2520*5, log_path = log_dir, best_model_save_path = log_dir) #change wehn script: set verbose to 0 so it doesn't waste time printing
        callback_1 = CustomEvalCallback(eval_env = eval_Env, eval_freq = 2520*5, log_path = log_dir_1, best_model_save_path = log_dir_1, verbose=0)
        #callback_2 = CustomEvalCallback(eval_env = eval_Env_Bench, eval_freq = 2520*5, log_path = log_dir_2, best_model_save_path = log_dir_2, verbose=0)
        
        #callback = SaveCumulativeRewardCallback(check_freq = 2520, log_dir = log_dir)
        
        years = 900
        
        model.learn(total_timesteps=(2520*years), callback = callback_1)
        

