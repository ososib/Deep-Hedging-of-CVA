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
#####################################################

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

###################################################################
###################################################################
def calculate_var(returns, percent):
    """
    Calculate the Value at Risk (VaR) given an array of returns and a confidence level.

    Parameters:
    - returns: numpy array of returns
    - confidence_level: float representing the confidence level (e.g., 0.01 for 99% confidence level)

    Returns:
    - Var: Value at Risk (VaR) at the given confidence level
    """
    # Sort the returns in ascending order
    #returns_sorted = np.sort(returns)

    # Compute the index corresponding to the quantile


    Var = np.percentile(returns, percent)

    return Var


def calculate_es(returns, alpha):

    # Sort the returns in ascending order
    #returns_sorted = np.sort(returns)

    # Compute the index corresponding to the quantile
    #Var = np.percentile(returns_sorted, percent)

      mu, std = stats.norm.fit(returns)
      ES = mu+(std*1/np.sqrt(2*np.pi)*np.exp(-0.5*stats.norm.ppf(alpha)**2))/(1-alpha)

      return ES

###################################################################
###################################################################

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
        return -calculate_es(self.dailypnl,alpha= 0.99)

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

###################################################################
###################################################################


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


apm = BOTH(dt = dt, rho=rho, lambda0=lambda0, kappa=kappa, mu=mu, nu=nu, jump_alpha=jump_alpha, jump_gamma=jump_gamma, r0=r0, sigma=sigma)
###################################################################
###################################################################

trading_cost = 0
treward = 7


env = HedgingEnv(apm, trading_cost_para=trading_cost, reward_function=treward, T=T, dt=dt, a=a, beta_i=beta_i)
eval_Env = HedgingEnv(apm, trading_cost_para=trading_cost, reward_function=treward, T=T, dt=dt, a=a, beta_i=beta_i)
eval_Env_Bench = HedgingEnv(apm, trading_cost_para=trading_cost, reward_function=1, T=T, dt=dt, a=a, beta_i=beta_i)


# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)


    
###################################################################
###################################################################

def replace_backslash_with_forward_slash(input_string):
    # Replace all backslashes with forward slashes
    return input_string.replace("\\", "/")

# Example usage:
    
path_dir = input ("Enter path :") 
path_dir = replace_backslash_with_forward_slash(path_dir)

#model = PPO.load("rho0tc05sqrt")
#model = SAC.load("ppo_try23")
model = PPO.load(path_dir)
#model = PPO.load("rho05tc05b")
rew=[0]

obs = env.reset()

obs = obs[0]
price=[obs[20]]
act=[obs[20]]


# Extracting self.Qs
Qs_extracted = obs[0:10]

# Extracting self.swaps
swaps_extracted = obs[10:20]

# Extracting the scalar from np.dot(self.Qs, self.swaps)
prica = obs[20]

# Extracting (1/2)*self.h
h_extracted = obs[21:]  # This should be the last 20 elements

i=0
while i<252:
    action, _states = model.predict(obs, deterministic=True)# model.predict(obs)
    obs, rewards, done, trunc, info = env.step(action)
    action = action
    price.append(obs[20])
    #act.append(action[0]+action[1]*obs[0]+action[2]*obs[1])
    dot_product = np.dot(action[:20], obs[:20])
    act.append(dot_product)
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


act=[0]
rew=[0]


#model = PPO.load("ppo_try2")


#########################################################################
#########################################################################
#adjust here!


obs, info = env.reset()  # Assuming reset returns a tuple of observation and info
price = [obs[20]]  # Assuming total price is stored here, adjust if needed

# Initialize arrays to store the values of A's and B's
A = obs[0:10]  # Assuming first 10 are A's
B = obs[10:20]  # Assuming next 10 are B's

# Arrays to track relative hedge weights
HA = [np.zeros(10)]  # Relative Hedge weights for A's
HB = [np.zeros(10)]  # Relative Hedge weights for B's

# Arrays to track absolute hedge weights
HAA = [] # Hedge weights for A's
HBB = [] # Hedge weights for B's   


Qs=[]
Swaps=[]

# Hedge performance tracking
hedgepl = [0]

i = 0
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    # Update hedge weights based on actions and current observations
    HA_current = action[:10] / B  # Update weights for A based on B values, handling zero division
    HB_current = action[10:] / A  # Update weights for B based on A values, handling zero division
    HA.append(HA_current)
    HB.append(HB_current)
    HAA.append(action[:10])  # Actual hedge actions for A's
    HBB.append(action[10:])  # Actual hedge actions for B's
    
    # Step in environment
    obs, rewards, done,trunc,info = env.step(action)
    price.append(obs[20])
    
    # Update A and B arrays with new observations
    A = obs[0:10]
    B = obs[10:20]
    
    # Compute hedge performance (simplified example, adjust accordingly)
    hedge_performance = obs[20] - np.dot(action[:10], A) - np.dot(action[10:], B)
    hedgepl.append(hedge_performance)
    
    Qs=[np.dot(action[:10], A)]
    Swaps=[np.dot(action[10:], B)]
    
    
    i += 1


total_hedge = []
for i in range(len(Swaps)):
    total_hedge.append(Qs[i] + Swaps[i])

plt.plot(Qs, label="Qs")
plt.plot(Swaps, label="Swaps")
plt.plot(total_hedge, label="Total hedge")
plt.plot(price, label="CVA(t)")
#plt.plot(HV)
plt.title("Market")
plt.legend()
plt.show()


# Visualization for A's and B's
plt.figure(figsize=(14, 7))
for idx in range(10):
    plt.plot([h[idx] for h in HA], label=f"A{idx}(t)")
    plt.plot([h[idx] for h in HB], label=f"B{idx}(t)")
plt.title("Relative Hedge Weights Over Time")
plt.legend()
plt.show()


# Sample size for demonstration; replace with actual length of your simulation
num_timesteps = len(HAA)  # Assuming HAA and HBB are updated each timestep

# Create figure and axes
fig, axs = plt.subplots(2, 10, figsize=(20, 10))  # 2 rows, 10 columns for A's and B's

# Set title
fig.suptitle('Actual Hedge Weights Over Time')

# Plot data for each A
for idx in range(10):
    axs[0, idx].plot([h[idx] for h in HAA], label=f"HAA{idx}")
    axs[0, idx].set_title(f"HAA{idx} (A{idx})")
    axs[0, idx].legend(loc="upper right")

# Plot data for each B
for idx in range(10):
    axs[1, idx].plot([h[idx] for h in HBB], label=f"HBB{idx}")
    axs[1, idx].set_title(f"HBB{idx} (B{idx})")
    axs[1, idx].legend(loc="upper right")

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to fit titles and suptitle
plt.show()



