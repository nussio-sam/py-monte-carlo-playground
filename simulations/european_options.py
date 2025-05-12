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
        #Also a function for european options, this checks for below strike price
        ST = self.GBM(sim_num)
        
        payoffs = np.maximum(self.strike_price - ST, np.zeros_like(ST))
            
        return e ** (-1 * self.rf_rate * self.time) * np.mean(payoffs)

def delta(self, bump_percentage: float, sim_num: int) -> float:
        #The delta of an option shows how much it changes as the underlying asset changes
        #There are closed formulas for this, but simulating with MC is more in line with this portfolio
        
        #The og_price is just the current option call price
        og_price = self.european_call(sim_num)
        
        #This calculates the call price if we bumped the initial price by a tiny bit
        bump_option = EuropeanOptions(self.strike_price, self.vol, self.rf_rate, self.time, self.init_price * (1 + bump_percentage))
        bumped_price = bump_option.european_call(sim_num)
        
        #Finding the output difference over the bump difference
        delta = (bumped_price - og_price) / (self.init_price * bump_percentage)
        return delta
