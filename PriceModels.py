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


class JDM(GenericAssetPriceModel):
    def __init__(self, s_0, r, sigma, dt, mu_J, lambda_J, sigma_J):
        self.dt = dt
        self.s_0 = s_0
        self.r = r
        self.sigma = sigma
        self.mu_J = mu_J
        self.lambda_J = lambda_J
        self.sigma_J = sigma_J
        self.current_price = s_0

    def compute_next_price(self):
        i = np.random.normal(0, np.sqrt(self.dt))
        j = np.random.poisson(self.dt*self.lambda_J)
        kappa = np.exp(self.mu_J) - 1;
        drift = self.r - self.lambda_J*(self.mu_J - self.sigma_J**2/2) - 0.5*self.sigma**2;
        jump = j*np.random.normal(self.mu_J - self.sigma_J**2/2,self.sigma_J)
        new_price = self.current_price * np.exp((drift) * self.dt
                   + self.sigma * i + jump)
        self.current_price = new_price

    def reset(self):
        self.current_price = self.s_0

    def get_current_price(self):
        return self.current_price


class HULL(GenericAssetPriceModel):
    def __init__(self, r_0, dt, sigma, alpha, a):
        self.current_r = r_0
        self.dt = dt
        self.sigma = sigma
        self.alpha = alpha
        self.a = a
        self.t = 0


    def compute_next_price(self, f):
        alpha = f + self.sigma**2/(2*self.a**2)*(1-np.exp(-self.a*self.t))**2
        e = self.current_r * np.exp(-self.a*self.dt) + alpha - self.alpha * np.exp(-self.a*self.dt)
        v = self.sigma**2/(2*self.a) * (1 - np.exp(-2*self.a*self.dt))
        r = np.random.normal(e, np.sqrt(v))
        self.alpha = alpha
        self.current_r = r
        self.t = self.t+self.dt

    def reset(self):
        self.current_r = self.r_0

    def get_current_price(self):
        return self.current_r
