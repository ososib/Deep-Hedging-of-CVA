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
from hedge_env import *
from callbacks import *
from esvar import *

import pandas as pd

df = pd.DataFrame({},
                  index=['Variance Total Cost', 'Variance Total Cost Without First', 'Mean Square Total Cost', 'Mean Total Cost','Value at Risk', 'Value at Risk Without First', 'Expected Shortfall', 'Expected Shortfall Without First', 'Mean Turnover'])

"""## Define Environment

p[2] = P(t) 
p[1] = A(t) 
p[0] = B(t) 
"""


rho = 0 #can be between -1 and 1!
trading_cost = 0.0



dt = 1/252
sigma_H = 0.03
r_H = 0.3

alpha = 0.1

mu_J = 0.2
s_0 = 0.4
sigma = 0.05
r = 0.01
lambda_J = 0.1
sigma_J = 0.5


apm = BOTH(r_0=r_H, dt = dt, sigma_H = sigma_H, alpha = alpha, s_0=s_0, r=r, sigma=sigma, mu_J=mu_J, lambda_J=lambda_J, sigma_J=sigma_J, rho=rho)

####################################################################################################################################################
####################################################################################################################################################

for treward in range(1,9):
        
    env = HedgingEnv(apm, trading_cost_para=trading_cost, reward_function=treward)
    eval_Env = HedgingEnv(apm, trading_cost_para=trading_cost, reward_function=treward)
    eval_Env_Bench = HedgingEnv(apm, trading_cost_para=trading_cost, reward_function=1)
    
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)
    
    
    # Create log dir
    log_dir_0 = "/home/osmanoscar/DH_mars/"+str(treward) + "/0/" # change wehn script to depend on reward function, rho, and tc
    os.makedirs(log_dir_0, exist_ok=True) #change exists?
    
    log_dir_1 = "/home/osmanoscar/DH_mars/"+str(treward) + "/1/" # change wehn script to depend on reward function, rho, and tc
    os.makedirs(log_dir_1, exist_ok=True) #change exists?
    
    log_dir_2 = "/home/osmanoscar/DH_mars/"+str(treward) + "/2/" # change wehn script to depend on reward function, rho, and tc
    os.makedirs(log_dir_2, exist_ok=True) #change exists?
    
    # Wrap the environment
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir_0)#, log_dir)
    eval_Env = Monitor(eval_Env, log_dir_1) #?
    eval_Env_Bench = Monitor(eval_Env_Bench, log_dir_2) #?
    
    """## Train agent"""
    
    model = PPO("MlpPolicy", env, verbose=0)
    #from stable_baselines3 import SAC
    #model = SAC("MlpPolicy", env, verbose=0) #performs badly. I think because it needs more hyperparameter tuning than PPO.
    
    #callback = EvalCallback(eval_env = eval_Env, eval_freq = 2520*5, log_path = log_dir, best_model_save_path = log_dir) #change wehn script: set verbose to 0 so it doesn't waste time printing
    callback_1 = CustomEvalCallback(eval_env = eval_Env, eval_freq = 2520*5, log_path = log_dir_1, best_model_save_path = log_dir_1, verbose=1)
    callback_2 = CustomEvalCallback(eval_env = eval_Env_Bench, eval_freq = 2520*5, log_path = log_dir_2, best_model_save_path = log_dir_2, verbose=0)
    
    #callback = SaveCumulativeRewardCallback(check_freq = 2520, log_dir = log_dir)
    
    years = 900
    
    model.learn(total_timesteps=(2520*years), callback = [callback_1, callback_2])
    
    
    #model.save("rho05tc05")
    
    #del model # remove to demonstrate saving and loading
    
    #model.save("rho05tc05")
    model = PPO.load("/home/osmanoscar/DH_mars/"+ str(treward)+"/1/best_model.zip")
    model.save("rho05tc05b")
    #model = PPO.load("rho05tc05") histdata = np.append(histdata[1:],newdata)
    
    
    '''
    def moving_average(values, window):
        """
        Smooth values by doing a moving average
        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, "valid")
    
    
    def plot_results(log_folder, title="Learning Curve"):
        """
        plot the results
    
        :param log_folder: (str) the save location of the results to plot
        :param title: (str) the title of the task to plot
        """
        x, y = ts2xy(load_results(log_folder), "timesteps")
        y = moving_average(y, window=5)
        # Truncate x
        x = x[len(x) - len(y) :]
    
        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
        plt.title(title + " Smoothed")
        plt.show()
    
    def plot_log_results(log_folder, title="Learning Curve"):
        """
        plot the results
    
        :param log_folder: (str) the save location of the results to plot
        :param title: (str) the title of the task to plot
        """
        x, y = ts2xy(load_results(log_folder), "timesteps")
        y = moving_average(y, window=5)
        # Truncate x
        x = x[len(x) - len(y) :]
    
        fig = plt.figure(title)
        plt.plot(x, np.log(-y))
        plt.xlabel("Number of Timesteps")
        plt.ylabel("log(-Rewards)")
        plt.title(title + " Smoothed")
        plt.show()
    
    plot_results(log_dir_0)
    plot_log_results(log_dir_0)
    plot_results(log_dir_1)
    plot_log_results(log_dir_1)
    plot_results(log_dir_2)
    plot_log_results(log_dir_2)
    '''
    
    """## Test Agent
    plots value of contract, value of hedge, and rewards the agent gets
    """
    
    '''
    model = PPO.load("rho05tc05b")
    
    rew=[0]
    
    obs = env.reset()
    
    obs = obs[0]
    price=[obs[2]]
    act=[obs[2]]
    
    i=0
    while i<252:
        action, _states = model.predict(obs, deterministic=True)# model.predict(obs)
        obs, rewards, done, trunc, info = env.step(action)
        action = 2*action
        price.append(obs[2])
        #act.append(action[0]+action[1]*obs[0]+action[2]*obs[1])
        act.append(action[0]*obs[0]+action[1]*obs[1])
        rew.append(rewards)
        i+=1
    
    plt.plot(price)
    plt.title("P(t)")
    plt.show()
    plt.plot(act[2:])
    plt.title("H(t)")
    plt.show()
    plt.plot(rew[2:])
    plt.title("Rewards(t)")
    plt.show()
    '''
    
    """plots:
    break-up of contract: how much is contract, how much is A, and how much is B
    break-up of hedge: how much in bank, how much in A and how much in B
    """
    
    '''
    obs = env.reset()
    obs = obs[0]
    price = [obs[2]]
    B=[obs[0]]
    A=[obs[1]]
    
    HA=[]
    HB=[]
    
    HAA=[]
    HBB=[]
    
    netportfolio=[]
    
    i=0
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action = 2*action
        HB.append(action[0]/obs[1])
        HA.append(action[1]/obs[0])
        HBB.append(action[0])
        HAA.append(action[1])
        obs, rewards, done, trunc, info = env.step(action)
        price.append(obs[2])
        act.append(action[0]*obs[0]+action[1]*obs[1])
        A.append(obs[1])
        B.append(obs[0])
        netportfolio.append(obs[2]-(action[0]*obs[0]+action[1]*obs[1]))
        i+=1
    
    
    plt.plot(A, label="A(t)")
    plt.plot(B, label="B(t)")
    plt.plot(price, label="P(t)")
    plt.title("Market")
    plt.legend()
    plt.show()
    
    plt.plot(HA, label="a(t)/B(t)")
    plt.plot(HB, label="b(t)/A(t)")
    plt.title("Relative hedge weights")
    plt.legend()
    plt.show()
    
    plt.plot(HAA, label="a(t)")
    plt.plot(HBB, label="b(t)")
    plt.title("Hedge weights")
    plt.legend()
    plt.show()
    
    plt.plot([x+y for (x,y) in zip(HA,HB)]) #==2 when delta hedging.
    plt.title("a(t)/B(t)+b(t)/A(t)")
    plt.show()
    
    plt.plot(price, label="CVA")
    plt.plot(act, label="-Hedge")
    plt.plot(netportfolio, label="Net")
    plt.title("Portfolio")
    plt.legend()
    plt.show()
    
    '''
    
    
    """## Model comparisons ##
    
    """
    
    Var_Total_Cost=[]
    Var_Total_Cost_Without_First=[]
    Mean_Square_Total_Cost=[]
    Mean_Total_Cost=[]
    Value_at_Risk=[]
    Value_at_Risk_Without_First=[]
    Expected_Shortfall=[]
    Expected_Shortfall_Without_First=[]
    Mean_Turnover=[]
    
    
    for i in range(10):
      obs = env.reset()
    
      obs = obs[0]
      acts = np.array([[0,0]])
    
      processes=np.array([obs[0:3]])
    
      done = False
      while not done:
          action, _states = model.predict(obs, deterministic=True)
          acts = np.append(acts, [2*action],0)
          obs, rewards, done, trunc, info = env.step(action)
          processes = np.append(processes,[obs[0:3]],0)
      acts = acts[1:,:]
    
      B=processes[:,0]
      A=processes[:,1]
      P=processes[:,2]
      Model_buy = B[:-1]*acts[:,0]+A[:-1]*acts[:,1]
      Model_sell = B[1:]*acts[:,0]+A[1:]*acts[:,1]
      TradingCost=trading_cost*(B[:-1]*np.abs(np.ediff1d(acts[:,0],to_begin=acts[0,0]))+A[:-1]*np.abs(np.ediff1d(acts[:,1],to_begin=acts[0,1])))
      HedgingError=np.ediff1d(P)-(Model_sell-Model_buy)
      TotalCost= HedgingError + TradingCost
      Var_Total_Cost.append(np.var(TotalCost))
      Var_Total_Cost_Without_First.append(np.var(TotalCost[1:]))
      Mean_Square_Total_Cost.append(np.mean(TotalCost**2))
      Mean_Total_Cost.append(np.mean(TotalCost))
      Value_at_Risk.append(calculate_var(TotalCost, 1))
      Value_at_Risk_Without_First.append(calculate_var(TotalCost[1:], 1))
      Expected_Shortfall.append(calculate_es(TotalCost,alpha= 0.01))
      Expected_Shortfall_Without_First.append(calculate_es(TotalCost,alpha= 0.01))
      netasset=P[1:]-Model_sell
      tradesB=B[:-1]*np.ediff1d(acts[:,0],to_begin=acts[0,0])
      tradesA=A[:-1]*np.ediff1d(acts[:,1],to_begin=acts[0,1])
      buyA = np.where(tradesA<0, tradesA, 0)
      buyB = np.where(tradesB<0, tradesB, 0)
      sellA = np.where(tradesA>0, tradesA, 0)
      sellB = np.where(tradesB>0, tradesB, 0)
      totalbuys = buyA + buyB
      totalsells = sellA + sellB
      Mean_Turnover.append(100*np.min([np.abs(np.sum(totalbuys)), np.abs(np.sum(totalsells))])/np.abs(np.mean(netasset)))
    
    df['Reward' + str(treward)]=[f'{np.mean(Var_Total_Cost):.2}', f'{np.mean(Var_Total_Cost_Without_First):.2}', f'{np.mean(Mean_Square_Total_Cost):.2}', f'{np.mean(Mean_Total_Cost):.2}', f'{np.mean(Value_at_Risk):.2}', f'{np.mean(Value_at_Risk_Without_First):.2}', f'{np.mean(Expected_Shortfall):.2}', f'{np.mean(Expected_Shortfall_Without_First):.2}', f'{np.mean(Mean_Turnover):.2}']

####################################################################################################################################################
####################################################################################################################################################

#when script, put outside parameter loop
## Do Delta:

Var_Total_Cost=[]
Var_Total_Cost_Without_First=[]
Mean_Square_Total_Cost=[]
Mean_Total_Cost=[]
Value_at_Risk=[]
Value_at_Risk_Without_First=[]
Expected_Shortfall=[]
Expected_Shortfall_Without_First=[]
Mean_Turnover=[]

for i in range(10):
  obs = env.reset()
  obs = obs[0]

  processes=np.array([obs[0:3]])

  nun=np.array([0,0])

  done = False
  while not done:
      obs, rewards, done, trunc, info = env.step(nun)
      processes = np.append(processes,[obs[0:3]],0)
  B=processes[:,0]
  A=processes[:,1]
  P=processes[:,2]
  Delta_buy = B[:-1]*A[:-1]+A[:-1]*B[:-1]
  Delta_sell = B[1:]*A[:-1]+A[1:]*B[:-1]
  TradingCost=trading_cost*(B[:-1]*np.abs(np.ediff1d(A[:-1],to_begin=A[0]))+A[:-1]*np.abs(np.ediff1d(B[:-1],to_begin=B[0])))
  HedgingError=np.ediff1d(P)-(Delta_sell-Delta_buy)
  TotalCost= HedgingError + TradingCost
  Var_Total_Cost.append(np.var(TotalCost))
  Var_Total_Cost_Without_First.append(np.var(TotalCost[1:]))
  Mean_Square_Total_Cost.append(np.mean(TotalCost**2))
  Mean_Total_Cost.append(np.mean(TotalCost))
  Value_at_Risk.append(calculate_var(TotalCost, 1))
  Value_at_Risk_Without_First.append(calculate_var(TotalCost[1:], 1))
  Expected_Shortfall.append(calculate_es(TotalCost,alpha= 0.01))
  Expected_Shortfall_Without_First.append(calculate_es(TotalCost,alpha= 0.01))
  netasset=P[1:]-Delta_sell
  tradesB=B[:-1]*np.ediff1d(A[:-1],to_begin=A[0])
  tradesA=A[:-1]*np.ediff1d(B[:-1],to_begin=B[0])
  buyA = np.where(tradesA<0, tradesA, 0)
  buyB = np.where(tradesB<0, tradesB, 0)
  sellA = np.where(tradesA>0, tradesA, 0)
  sellB = np.where(tradesB>0, tradesB, 0)
  totalbuys = buyA + buyB
  totalsells = sellA + sellB
  Mean_Turnover.append(100*np.min([np.abs(np.sum(totalbuys)), np.abs(np.sum(totalsells))])/np.abs(np.mean(netasset)))


df['Delta']=[f'{np.mean(Var_Total_Cost):.2}', f'{np.mean(Var_Total_Cost_Without_First):.2}', f'{np.mean(Mean_Square_Total_Cost):.2}', f'{np.mean(Mean_Total_Cost):.2}', f'{np.mean(Value_at_Risk):.2}', f'{np.mean(Value_at_Risk_Without_First):.2}', f'{np.mean(Expected_Shortfall):.2}', f'{np.mean(Expected_Shortfall_Without_First):.2}', f'{np.mean(Mean_Turnover):.2}']


## Do Nothing:

Var_Total_Cost=[]
Var_Total_Cost_Without_First=[]
Mean_Square_Total_Cost=[]
Mean_Total_Cost=[]
Value_at_Risk=[]
Value_at_Risk_Without_First=[]
Expected_Shortfall=[]
Expected_Shortfall_Without_First=[]
Mean_Turnover=[]

for i in range(10):
  obs = env.reset()
  obs = obs[0]

  processes=np.array([obs[0:3]])

  nun=np.array([0,0])

  done = False
  while not done:
      obs, rewards, done, trunc, info = env.step(nun)
      processes = np.append(processes,[obs[0:3]],0)
  B=processes[:,0]
  A=processes[:,1]
  P=processes[:,2]
  Delta_buy = 0
  Delta_sell = 0
  TradingCost= 0
  HedgingError=np.ediff1d(P)-(Delta_sell-Delta_buy)
  TotalCost= HedgingError + TradingCost
  Var_Total_Cost.append(np.var(TotalCost))
  Var_Total_Cost_Without_First.append(np.var(TotalCost[1:]))
  Mean_Square_Total_Cost.append(np.mean(TotalCost**2))
  Mean_Total_Cost.append(np.mean(TotalCost))
  Value_at_Risk.append(calculate_var(TotalCost, 1))
  Value_at_Risk_Without_First.append(calculate_var(TotalCost[1:], 1))
  Expected_Shortfall.append(calculate_es(TotalCost,alpha= 0.01))
  Expected_Shortfall_Without_First.append(calculate_es(TotalCost,alpha= 0.01))
  netasset=P[1:]-Delta_sell
  tradesB=0
  tradesA=0
  buyA = np.where(tradesA<0, tradesA, 0)
  buyB = np.where(tradesB<0, tradesB, 0)
  sellA = np.where(tradesA>0, tradesA, 0)
  sellB = np.where(tradesB>0, tradesB, 0)
  totalbuys = buyA + buyB
  totalsells = sellA + sellB
  Mean_Turnover.append(100*np.min([np.abs(np.sum(totalbuys)), np.abs(np.sum(totalsells))])/np.abs(np.mean(netasset)))

df['Nothing']=[f'{np.mean(Var_Total_Cost):.2}', f'{np.mean(Var_Total_Cost_Without_First):.2}', f'{np.mean(Mean_Square_Total_Cost):.2}', f'{np.mean(Mean_Total_Cost):.2}', f'{np.mean(Value_at_Risk):.2}', f'{np.mean(Value_at_Risk_Without_First):.2}', f'{np.mean(Expected_Shortfall):.2}', f'{np.mean(Expected_Shortfall_Without_First):.2}', f'{np.mean(Mean_Turnover):.2}']

(df.T).to_csv('PerformanceTable.csv')
