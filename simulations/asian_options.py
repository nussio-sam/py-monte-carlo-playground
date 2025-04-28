import numpy as np
from american_options import AmericanOptions

class AsianOptions(AmericanOptions):
    '''
    Asian Options are a type of exotic option using the average price
    of the underlying asset over the course of a set of discrete timestamps
    to determine the payoff strike price
    '''
    def __init__(self, vol, rf_rate, time, init_price, timesteps = 1, strike_price = 0):
        super().__init__(strike_price, vol, rf_rate, time, init_price)
        #Timesteps = The amount of steps of time calculated between contract start and expiry
        #which can be cashed out at
        #This can be tuned to find the discrete timestamps to check payout at
        self.timesteps = timesteps
        self.parameters['Timesteps'] = self.timesteps
        
    @staticmethod    
    def arithmetic_mean(array):
        #Using numpy functions to create a 1d array of arithmetic means from a 2d input array
        #The 2d input array will be the output of timestep_GBM
        sums = np.sum(array, axis = 1)
        return sums /(np.size(array, 1))
    
    @staticmethod
    def geometric_mean(array):
        products = np.prod(array, axis = 1)
        return (products) ** (1 / np.size(array, 1))
    
    def asian_call(self, mean_func, sim_num: int):
        #Finding the mean call profit of an asian option, with a parameter for mean function
        #This allows us to define the difference between a geometric mean and arithmetic mean
        #Without creating separate call and put functions for each mean
        
        #We still need paths like american_options because we need to find the mean over time
        #even if we arent deciding at each timestep
        paths = self.timestep_GBM(sim_num)
        
        #This 1d array will pass the mean function through to apply onto our paths
        means = mean_func(paths)
        disc = np.exp(-1 * self.rf_rate * self.time)
        
        #Finding the payoffs as either the final price minus mean, or zero
        #Very similar to european call, just with a different strike price
        payoffs = np.maximum(means - self.strike_price, np.zeros_like(means))
       
        #returning discounted payoffs
        return disc * np.mean(payoffs)
    
    def asian_put(self, mean_func, sim_num: int):
        #The asian put is similar to the asian call, just with a reversed payoff formula
        #All lines here correlate to the comments of asian_call
        
        paths = self.timestep_GBM(sim_num)
        means = mean_func(paths)
        disc = np.exp(-1 * self.rf_rate * self.time)
        payoffs = np.maximum(self.strike_price - means, np.zeros_like(means))
        return disc * np.mean(payoffs)
    
asian = AsianOptions(0.2, 0.04, 1, 100, timesteps = 50, strike_price = 105)
print(asian.asian_call(AsianOptions.arithmetic_mean, 1000))
print(asian.asian_put(AsianOptions.arithmetic_mean, 1000))
print(asian.asian_call(AsianOptions.geometric_mean, 1000))
print(asian.asian_put(AsianOptions.geometric_mean, 1000))
        
        
        