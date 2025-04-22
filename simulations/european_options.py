import numpy as np
from options import Options

class EuropeanOptions(Options):
    '''
    Simulates European Option call/put end pricing
    using Monte Carlo simulations
    '''
    
    def __init__(self, strike_price, vol, rf_rate, time, init_price):
        super().__init__(strike_price, vol, rf_rate, time, init_price)
        
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