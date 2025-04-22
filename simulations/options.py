import numpy as np
from math import e

# The Geometric Brownian Motion is the first step towards modeling the full Black-Scholes Model
# Using Monte-Carlo Approximations/simulations.
# This random motion, where the logarithm of the quantity follows Brownian Motion,
# is used to simulate stock prices

class Options:
    
    def __init__(self, strike_price: float, vol: float, rf_rate: float, time: float, init_price: float):
        
        #These are all the basic variables in the GBM equation for option pricing
        '''
        Initial Price= The price of the option at the start date of the contract
        Strike Price = The price determined in contract that the option can be bought or sold at
        Volatility = The option's susceptibility to change based on the underlying asset
        Risk-free Rate = The rate of money increase of a risk-free contract (based on Treasury Yield usually)
        Time to Maturity = How long until the contract expires
        '''
        self.strike_price = strike_price
        self.vol = vol
        self.rf_rate = rf_rate
        self.time = time
        self.init_price = init_price
        self.parameters = {
            'Strike price' : self.strike_price,
            'Volatility' : self.vol,
            'Risk-free Rate' : self.rf_rate,
            'Time to Maturity' : self.time,
            'Initial Price' : self.init_price,
            }
        
        self.validate()
        
    def validate(self):
        
        for key, val in self.parameters.items():
            if not isinstance(val, (float, int)):
                print(f'{key} is not a number!')
            if key != 'Risk-free Rate' and val <= 0:
                print(f'{key} is less than 0.')
        
        
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
        