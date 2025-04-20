from random import gauss as normal
from math import e

# The Geometric Brownian Motion is the first step towards modeling the full Black-Scholes Model
# Using Monte-Carlo Approximations/simulations.
# This random motion, where the logarithm of the quantity follows Brownian Motion,
# is used to simulate stock prices

class european_options:
    
    def __init__(self, strike_price: int, vol: int, rf_rate: int, time: int, init_price: int, time_steps = 1):
        
        #These are all the basic variables in the GBM equation for option pricing
        self.strike_price = strike_price
        self.vol = vol
        self.rf_rate = rf_rate
        self.time = time
        self.init_price = init_price
        self.time_steps = time_steps
        
    def GBM(self, sim_num):
        # The discrete (analytic) form, the equation goes by timestep, solved through Itô Calculus
        Z = []
        for i in range(sim_num):
            Z.append(normal(mu = 0.0, sigma = 1))
        
        #Initializing and iterating over the end result, which is a dimensional list the same size as our sim_num
        #This list is the end of period stock pricings
        ST = []
        for rand in Z:
            ST.append(self.init_price * e**(self.time * (self.rf_rate - (0.5 * self.vol ** 2)) + self.vol * self.time**0.5 * rand))
        
        return ST
        
    def european_call(self, sim_num: int) -> float:
        #As a function for european options, this only checks for price by end of maturity time
        
        #Using the GBM to simulate stock paths
        ST = self.GBM(sim_num)
        
        #Simply finding if the payoff is worth it or not, placing 0 if not
        payoffs = []
        for end_price in ST:
            payoffs.append(max(end_price - self.strike_price, 0))
            
        return e ** ( -1 * self.rf_rate * self.time) * (sum(payoffs)/len(payoffs))
    
    def european_put(self, sim_num: int) -> float:
        #Also a function for european options, this checks for below strike price
        ST = self.GBM(sim_num)
        
        payoffs = []
        for end_price in ST:
            payoffs.append(max(self.strike_price - end_price, 0))
            
        return e ** (-1 * self.rf_rate * self.time) * (sum(payoffs)/len(payoffs))

european = european_options(105, 0.2, 0.01, 1, 100)
print(european.european_call(100))
print(european.european_put(100))
