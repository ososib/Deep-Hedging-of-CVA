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

Qs = 0:9
Swaps=10:19
cva = 20
Q_quantity=21:30
Swaps_quantity=31:40
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
  

for evals in [2]:

  for trading_cost in [0]:# [0, 0.05, 1]:

    for rho in [-1,-0.5,0,0.5,1]:
        
        df = pd.DataFrame({},
                          index=['Variance of Losses', 'Variance of Losses$^{*}$', 'Mean Square Losses', 'Mean Losses','Value at Risk', 'Expected Shortfall', 'Turnover'])
        for rewd in range(1,8):
          df['Reward ' + str(rewd)]= [0, 0, 0, 0, 0, 0, 0]
        df['Delta']= [0, 0, 0, 0, 0, 0, 0]
        df['Nothing']= [0, 0, 0, 0, 0, 0, 0]
        
        
        
        apm = BOTH(dt = dt, rho=rho, lambda0=lambda0, kappa=kappa, mu=mu, nu=nu, jump_alpha=jump_alpha, jump_gamma=jump_gamma, r0=r0, sigma=sigma)


        


        env = HedgingEnv(apm, trading_cost_para=trading_cost, reward_function=treward, T=T, dt=dt, a=a, beta_i=beta_i)
        
        for reals in range(0,100):
          
          
          obs = env.reset()
          obs = obs[0]
          
          Qs = obs[0:10] 
          Swaps = obs[10:20]
          cva = obs[20]
          Q_quantity = obs[21:31]
          Swaps_quantity = obs[31:]
          
          processes=np.array([obs[0:3]])
          nun=np.array([0]*20)
          done = False
          
          
          while not done:
            obs, rewards, done, trunc, info = env.step(nun)
            processes = np.append(processes,[obs[0:20]],0) #processes is a matrix with the first row being the initial state and the rest being the states at each time step, the columns are Qs, Swaps, and cva 
            
        

          ####################################################################################################################################################
          ####################################################################################################################################################
          
          for treward in [7]: #range(1,8):
            env = HedgingEnv(apm, trading_cost_para=trading_cost, reward_function=treward)
            model = PPO.load("/home/osmanoscar/CVA_results_04_29/tmp/"+ 'tc' + str(trading_cost) +'/rho'+str(rho)+'/'+ 'rwf' +str(treward)+"/" + str(evals) + "/best_model.zip")
            
            """## Model comparisons ##
            """

          
            acts_Q = np.array([[0]*10])
            acts_S = np.array([[0]*10])
            action = np.array([[0]*20])

            for i in range(len(processes)):
              obs = processes[i]
              action, _states = model.predict(np.append(obs, [action]), deterministic=True)
              acts = np.append(acts, [action],0)
            
            acts = acts[1:-1,:]
            
            Qs = processes[:, 0:10]
            Swaps = processes[:, 10:20]
            P = processes[:, 20]
            
            rows, columns = Qs.shape
            zero_row=np.zeros(1,columns)

            Model_buy = np.sum(Qs[:-1] * acts[:, 0:10], axis=1) + np.sum(Swaps[:-1] * acts[:, 10:20], axis=1)
            Model_sell = np.sum(Qs[1:] * acts[:, 0:10], axis=1) + np.sum(Swaps[1:] * acts[:, 10:20], axis=1)

            TradingCost = trading_cost * (np.sum(  Qs[:-1] * np.diff(acts[:, 0:10],axis=0, prepend=zero_row), axis=1) + np.sum(Swaps[:-1] * np.diff(acts[:, 10:20], axis=0, prepend=zero_row), axis=1 ) )

            HedgingError = np.ediff1d(P) - (Model_sell - Model_buy)
            TotalCost = HedgingError - TradingCost

            Var_Total_Cost = (np.var(TotalCost))
            Var_Total_Cost_Without_First = (np.var(TotalCost[1:]))
            Mean_Square_Total_Cost = (np.mean(TotalCost**2))
            Mean_Total_Cost = (np.mean(-TotalCost))
            Value_at_Risk=(calculate_var(-TotalCost, 99))
            Expected_Shortfall=(calculate_es(-TotalCost,alpha= 0.99))

            netasset = P[1:] - Model_sell

            tradesQs = np.sum(  Qs[:-1] * np.diff(acts[:, 0:10],axis=0, prepend=zero_row), axis=1) 
            tradesSwaps = np.sum(Swaps[:-1] * np.diff(acts[:, 10:20], axis=0, prepend=zero_row), axis=1) 

            buyQs = np.where(tradesQs < 0, tradesQs, 0)
            buySwaps = np.where(tradesSwaps < 0, tradesSwaps, 0)
            sellQs = np.where(tradesQs > 0, tradesQs, 0)
            sellSwaps = np.where(tradesSwaps > 0, tradesSwaps, 0)

            totalbuys = np.sum(buyQs, axis=0) + np.sum(buySwaps, axis=0)
            totalsells = np.sum(sellQs, axis=0) + np.sum(sellSwaps, axis=0)

            Mean_Turnover = (100 * np.min([np.abs(np.sum(totalbuys)), np.abs(np.sum(totalsells))]) / np.abs(np.mean(netasset)))
            
            df['Reward ' + str(treward)]=df['Reward ' + str(treward)]+[x/100 for x in [Var_Total_Cost, Var_Total_Cost_Without_First, Mean_Square_Total_Cost, Mean_Total_Cost, Value_at_Risk, Expected_Shortfall, Mean_Turnover]]
            
          ####################################################################################################################################################
          ####################################################################################################################################################
          

          ## Do Delta:
          rows, columns = Qs.shape
          zero_row=np.zeros(1,columns)
        
          Qs = processes[:, 0:10] #This operation selects all rows (:) and the first 10 columns (0:10) of the array
          Swaps = processes[:, 10:20]
          P = processes[:, 20]
          
          Delta_buy =   np.sum(Qs[:-1] * Swaps[:-1], axis=1) + np.sum(Swaps[:-1] * Qs[:-1], axis=1) # B[:-1]*A[:-1]+A[:-1]*B[:-1]
          Delta_sell =   np.sum(Qs[1:] * Swaps[:-1], axis=1) + np.sum(Swaps[1:] * Qs[:-1], axis=1) # B[1:]*A[:-1]+A[1:]*B[:-1]
          
          
          TradingCost=trading_cost*np.sum(Qs[:-1]*np.diff(Swaps[:-1], axis=0, prepend=zero_row), axis=1)+np.sum(Swaps[:-1]*np.diff(Qs[:-1], axis=0, prepend=zero_row), axis=1) 
          #TradingCost = trading_cost * (np.sum(Qs[:-1] * np.abs(np.ediff1d(acts[:, 0:10], to_begin=acts[0, 0:10])), axis=1) + np.sum(Swaps[:-1] * np.abs(np.ediff1d(acts[:, 10:20], to_begin=acts[0, 10:20])), axis=1))
          #TradingCost=trading_cost*(B[:-1]*np.abs(np.ediff1d(A[:-1],to_begin=A[0]))+A[:-1]*np.abs(np.ediff1d(B[:-1],to_begin=B[0])))
          
          HedgingError=np.ediff1d(P)-(Delta_sell-Delta_buy)
          TotalCost= HedgingError - TradingCost

          Var_Total_Cost=(np.var(TotalCost))
          Var_Total_Cost_Without_First=(np.var(TotalCost[1:]))
          Mean_Square_Total_Cost=(np.mean(TotalCost**2))
          Mean_Total_Cost=(np.mean(-TotalCost))
          Value_at_Risk=(calculate_var(-TotalCost, 99))
          Expected_Shortfall=(calculate_es(-TotalCost,alpha= 0.99))
          netasset=P[1:]-Delta_sell
          tradesB= np.sum(Qs[:-1]*np.diff(Swaps[:-1], axis=0, prepend=zero_row), axis=1)
          tradesA= np.sum(Swaps[:-1]*np.diff(Qs[:-1], axis=0, prepend=zero_row), axis=1)
          buyA = np.where(tradesA<0, tradesA, 0)
          buyB = np.where(tradesB<0, tradesB, 0)
          sellA = np.where(tradesA>0, tradesA, 0)
          sellB = np.where(tradesB>0, tradesB, 0)
          totalbuys = buyA + buyB
          totalsells = sellA + sellB
          Mean_Turnover=(100*np.min([np.abs(np.sum(totalbuys)), np.abs(np.sum(totalsells))])/np.abs(np.mean(netasset)))

          df['Delta']=df['Delta']+[x/100 for x in [Var_Total_Cost, Var_Total_Cost_Without_First, Mean_Square_Total_Cost, Mean_Total_Cost, Value_at_Risk, Expected_Shortfall, Mean_Turnover]]
          
          ## Do Nothing:
          
          P = processes[:, 20]
          HedgingError=np.ediff1d(P)
          TotalCost= HedgingError

          Var_Total_Cost=(np.var(TotalCost))
          Var_Total_Cost_Without_First=(np.var(TotalCost[1:]))
          Mean_Square_Total_Cost=(np.mean(TotalCost**2))
          Mean_Total_Cost=(np.mean(-TotalCost))
          Value_at_Risk=(calculate_var(-TotalCost, 99))
          Expected_Shortfall=(calculate_es(-TotalCost,alpha= 0.99))
          
          Mean_Turnover=0
          
          df['Nothing']= df['Nothing'] + [x/100 for x in [Var_Total_Cost, Var_Total_Cost_Without_First, Mean_Square_Total_Cost, Mean_Total_Cost, Value_at_Risk, Expected_Shortfall, Mean_Turnover]]
          
        #(df.T).to_latex("/home/osmanoscar/DH_mars/tmp/" + 'rho' + str(rho) + '/' + 'PerformanceTable' + str(rho) + '.tex', float_format="%.2E", index = True, caption = "Average performance metrics over 10 realizations with increment correlation $\\rho=" + str(rho) + "$ and trading cost $tc=0.00$." + "{\\tiny$^*$ excluding first point.}", label = "PerfMat" + str(rho) + "tc000")
        (df.T).style.format({'Variance of Losses':'{:.2E}', 'Variance of Losses$^{*}$':'{:.2E}', 'Mean Square Losses':'{:.2E}', 'Mean Losses':'{:.2E}', 'Value at Risk':'{:.2E}', 'Expected Shortfall':'{:.2E}', 'Turnover':'{:.2E}'}).highlight_min(axis=0, props="textbf:--rwrap;").to_latex("/home/osmanoscar/DH_mars/tmp/"+ 'tc' + str(trading_cost) + '/'+ str(evals) + 'PerformanceTable' + 'rho' + str(rho) + 'tc' + str(trading_cost) + '.tex', caption = "Average performance metrics over 100 realizations with increment correlation $\\rho=" + str(rho) + "$ and trading cost $c=" + str(trading_cost) + "$." + "{\\tiny$^*$ excluding first point.}", label = "PerfMat" + str(rho) + "tc" + str(trading_cost), hrules=True)


