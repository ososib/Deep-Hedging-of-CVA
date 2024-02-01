import numpy as np
from abc import ABC, abstractmethod

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
        #self.r = self.current_r
        self.rho = rho


    def compute_next_price(self, theta):
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
        x3 = self.rho*x1+np.sqrt(1-self.rho**2)*x2
        i = np.sqrt(self.dt)*x1
        i2 = np.sqrt(self.dt)*x3
        #i = np.random.normal(0, np.sqrt(self.dt))
        j = np.random.poisson(self.dt*self.lambda_J)
        kappa = np.exp(self.mu_J) - 1;
        drift = self.r - self.lambda_J*(self.mu_J - self.sigma_J**2/2) - 0.5*self.sigma**2;
        jump = j*np.random.normal(self.mu_J - self.sigma_J**2/2,self.sigma_J)
        new_price = self.current_price * np.exp(drift * self.dt + self.sigma * i + jump)
        self.current_price = new_price
        self.current_r = self.current_r + (theta-self.alpha*self.current_r)*self.dt+self.sigma_H*i2#np.random.normal(0, np.sqrt(self.dt))
        #self.r = self.current_r

    def reset(self):
        self.current_r = self.r_0
        self.current_price = self.s_0

    def get_current_price(self):
        return self.current_r, self.current_price
