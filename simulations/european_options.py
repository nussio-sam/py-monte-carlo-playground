import numpy as np
from math import e

# The Geometric Brownian Motion is the first step towards modeling the full Black-Scholes Model
# Using Monte-Carlo Approximations/simulations.
# This random motion, where the logarithm of the quantity follows Brownian Motion,
# is used to simulate stock prices

class EuropeanOptions:
    
    def __init__(self, strike_price: float, vol: float, rf_rate: float, time: float, init_price: float, time_steps = 1):
        
        #These are all the basic variables in the GBM equation for option pricing
        self.strike_price = strike_price
        self.vol = vol
        self.rf_rate = rf_rate
        self.time = time
        self.init_price = init_price
        self.time_steps = time_steps
        
    def GBM(self, sim_num: float):
        # Simulates final stock prices using the analytic solution of GBM
        
        #Creating array of random normal numbers with sim_num size
        Z = np.random.normal(0, 1, sim_num)
        
        #Dividing up the GBM equation into manageable chunks of drift and diffusion
        drift = self.time * (self.rf_rate - (0.5 * self.vol ** 2))
        diff = self.vol * np.sqrt(self.time) * Z

        #Simulate end-of-period stock prices (ST) based on normal samples
        ST = self.init_price * np.exp(drift + diff)
        return ST
        
    def european_call(self, sim_num: int) -> float:
        #As a function for european options, this only checks for price by end of maturity time
        
        #Using the GBM to simulate stock paths
        ST = self.GBM(sim_num)
        
        #Calculating payoffs of end-of-period call through max of paths
        payoffs = np.maximum(ST - self.strike_price, np.zeros_like(ST))
            
        #returning the discounted payoffs
        return e ** ( -1 * self.rf_rate * self.time) * np.mean(payoffs)
    
    def european_put(self, sim_num: int) -> float:
        #Also a function for american options, this checks for below strike price
        ST = self.GBM(sim_num)
        
        payoffs = np.maximum(self.strike_price - ST, np.zeros_like(ST))
            
        return e ** (-1 * self.rf_rate * self.time) * np.mean(payoffs)
            
        return e ** (-1 * self.rf_rate * self.time) * (sum(payoffs)/len(payoffs))

european = european_options(105, 0.2, 0.01, 1, 100)
print(european.european_call(100))
print(european.european_put(100))
