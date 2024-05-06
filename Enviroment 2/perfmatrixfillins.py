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

"""## Define Environment

p[2] = P(t) 
p[1] = A(t) 
p[0] = B(t) 
"""


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

for evals in [2]:

  for trading_cost in [0.05]:

    for rho in [-1,-0.5,0,0.5,1]:
        
        df = pd.DataFrame({},
                          index=['Variance of Losses', 'Variance of Losses$^{*}$', 'Mean Square Losses', 'Mean Losses','Value at Risk', 'Expected Shortfall', 'Turnover'])
        for rewd in range(1,8):
          df['Reward ' + str(rewd)]= [0, 0, 0, 0, 0, 0, 0]
        df['Delta']= [0, 0, 0, 0, 0, 0, 0]
        df['Nothing']= [0, 0, 0, 0, 0, 0, 0]
        
        
        
        apm = BOTH(r_0=r_H, dt = dt, sigma_H = sigma_H, alpha = alpha, s_0=s_0, r=r, sigma=sigma, mu_J=mu_J, lambda_J=lambda_J, sigma_J=sigma_J, rho=rho)
        


        env = HedgingEnv(apm, trading_cost_para=trading_cost, reward_function=1)
        for reals in range(0,100):
          obs = env.reset()
          obs = obs[0]
          processes=np.array([obs[0:3]])
          nun=np.array([0,0])
          done = False
          while not done:
            obs, rewards, done, trunc, info = env.step(nun)
            processes = np.append(processes,[obs[0:3]],0)

          ####################################################################################################################################################
          ####################################################################################################################################################
          
          for treward in range(1,8):
            env = HedgingEnv(apm, trading_cost_para=trading_cost, reward_function=treward)
            model = PPO.load("/home/osmanoscar/DH_april/tmp/"+ 'tc' + str(trading_cost) +'/rho'+str(rho)+'/'+ 'rwf' +str(treward)+"/" + str(evals) + "/best_model.zip")
            
            """## Model comparisons ##
            """

          
            acts = np.array([[0,0]])
            action = np.array([[0,0]])

            for i in range(len(processes)):
              obs = processes[i]
              action, _states = model.predict(np.append(obs, [2*action]), deterministic=True)
              acts = np.append(acts, [2*action],0)
            acts = acts[1:-1,:]
          
            B=processes[:,0]
            A=processes[:,1]
            P=processes[:,2]
            Model_buy = B[:-1]*acts[:,0]+A[:-1]*acts[:,1]
            Model_sell = B[1:]*acts[:,0]+A[1:]*acts[:,1]
            TradingCost=trading_cost*(B[:-1]*np.abs(np.ediff1d(acts[:,0],to_begin=acts[0,0]))+A[:-1]*np.abs(np.ediff1d(acts[:,1],to_begin=acts[0,1])))
            HedgingError=np.ediff1d(P)-(Model_sell-Model_buy)
            TotalCost= HedgingError - TradingCost


            Var_Total_Cost=(np.var(TotalCost))
            Var_Total_Cost_Without_First=(np.var(TotalCost[1:]))
            Mean_Square_Total_Cost=(np.mean(TotalCost**2))
            Mean_Total_Cost=(np.mean(-TotalCost))
            Value_at_Risk=-(calculate_var(TotalCost, 1))
            Expected_Shortfall=(calculate_es(-TotalCost,alpha= 0.99))
            netasset=P[1:]-Model_sell
            tradesB=B[:-1]*np.ediff1d(acts[:,0],to_begin=acts[0,0])
            tradesA=A[:-1]*np.ediff1d(acts[:,1],to_begin=acts[0,1])
            buyA = np.where(tradesA<0, tradesA, 0)
            buyB = np.where(tradesB<0, tradesB, 0)
            sellA = np.where(tradesA>0, tradesA, 0)
            sellB = np.where(tradesB>0, tradesB, 0)
            totalbuys = buyA + buyB
            totalsells = sellA + sellB
            Mean_Turnover=(100*np.min([np.abs(np.sum(totalbuys)), np.abs(np.sum(totalsells))])/np.abs(np.mean(netasset)))
            
            df['Reward ' + str(treward)]=df['Reward ' + str(treward)]+[x/100 for x in [Var_Total_Cost, Var_Total_Cost_Without_First, Mean_Square_Total_Cost, Mean_Total_Cost, Value_at_Risk, Expected_Shortfall, Mean_Turnover]]
            
          ####################################################################################################################################################
          ####################################################################################################################################################
          

          ## Do Delta:
        
          B=processes[:,0]
          A=processes[:,1]
          P=processes[:,2]
          Delta_buy = B[:-1]*A[:-1]+A[:-1]*B[:-1]
          Delta_sell = B[1:]*A[:-1]+A[1:]*B[:-1]
          TradingCost=trading_cost*(B[:-1]*np.abs(np.ediff1d(A[:-1],to_begin=A[0]))+A[:-1]*np.abs(np.ediff1d(B[:-1],to_begin=B[0])))
          HedgingError=np.ediff1d(P)-(Delta_sell-Delta_buy)
          TotalCost= HedgingError - TradingCost

          Var_Total_Cost=(np.var(TotalCost))
          Var_Total_Cost_Without_First=(np.var(TotalCost[1:]))
          Mean_Square_Total_Cost=(np.mean(TotalCost**2))
          Mean_Total_Cost=(np.mean(-TotalCost))
          Value_at_Risk=-(calculate_var(TotalCost, 1))
          Expected_Shortfall=(calculate_es(-TotalCost,alpha= 0.99))
          netasset=P[1:]-Delta_sell
          tradesB=B[:-1]*np.ediff1d(A[:-1],to_begin=A[0])
          tradesA=A[:-1]*np.ediff1d(B[:-1],to_begin=B[0])
          buyA = np.where(tradesA<0, tradesA, 0)
          buyB = np.where(tradesB<0, tradesB, 0)
          sellA = np.where(tradesA>0, tradesA, 0)
          sellB = np.where(tradesB>0, tradesB, 0)
          totalbuys = buyA + buyB
          totalsells = sellA + sellB
          Mean_Turnover=(100*np.min([np.abs(np.sum(totalbuys)), np.abs(np.sum(totalsells))])/np.abs(np.mean(netasset)))

          df['Delta']=df['Delta']+[x/100 for x in [Var_Total_Cost, Var_Total_Cost_Without_First, Mean_Square_Total_Cost, Mean_Total_Cost, Value_at_Risk, Expected_Shortfall, Mean_Turnover]]
          
          ## Do Nothing:
          
          B=processes[:,0]
          A=processes[:,1]
          P=processes[:,2]
          HedgingError=np.ediff1d(P)
          TotalCost= HedgingError

          Var_Total_Cost=(np.var(TotalCost))
          Var_Total_Cost_Without_First=(np.var(TotalCost[1:]))
          Mean_Square_Total_Cost=(np.mean(TotalCost**2))
          Mean_Total_Cost=(np.mean(-TotalCost))
          Value_at_Risk=-(calculate_var(TotalCost, 1))
          Expected_Shortfall=(calculate_es(-TotalCost,alpha= 0.99))
          
          Mean_Turnover=0
          
          df['Nothing']= df['Nothing'] + [x/100 for x in [Var_Total_Cost, Var_Total_Cost_Without_First, Mean_Square_Total_Cost, Mean_Total_Cost, Value_at_Risk, Expected_Shortfall, Mean_Turnover]]
          
        #(df.T).to_latex("/home/osmanoscar/DH_mars/tmp/" + 'rho' + str(rho) + '/' + 'PerformanceTable' + str(rho) + '.tex', float_format="%.2E", index = True, caption = "Average performance metrics over 10 realizations with increment correlation $\\rho=" + str(rho) + "$ and trading cost $tc=0.00$." + "{\\tiny$^*$ excluding first point.}", label = "PerfMat" + str(rho) + "tc000")
        (df.T).style.format({'Variance of Losses':'{:.2E}', 'Variance of Losses$^{*}$':'{:.2E}', 'Mean Square Losses':'{:.2E}', 'Mean Losses':'{:.2E}', 'Value at Risk':'{:.2E}', 'Expected Shortfall':'{:.2E}', 'Turnover':'{:.2E}'}).highlight_min(axis=0, props="textbf:--rwrap;").to_latex("/home/osmanoscar/DH_april/tmp/"+ 'tc' + str(trading_cost) + '/'+ str(evals) + 'PerformanceTable' + 'rho' + str(rho) + 'tc' + str(trading_cost) + '.tex', caption = "Average performance metrics over 100 realizations with increment correlation $\\rho=" + str(rho) + "$ and trading cost $c=" + str(trading_cost) + "$." + "{\\tiny$^*$ excluding first point.}", label = "PerfMat" + str(rho) + "tc" + str(trading_cost), hrules=True)


