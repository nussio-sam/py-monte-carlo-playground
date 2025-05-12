import numpy as np
from options import Options

class AmericanOptions(Options):
    '''
    Simulating American Options paths through Geometric Brownian Motion
    and vectorization using numpy
    MERICUH
    '''
    
    def __init__(self, strike_price, vol, rf_rate, time, init_price, timesteps = 1):
        super().__init__(strike_price, vol, rf_rate, time, init_price)
        self.timesteps = timesteps
        #Timesteps = The amount of steps of time calculated between contract start and expiry
        #which can be cashed out at
        #This can be tuned for exotic options that only allow cashout at predetermined times
        self.parameters['Timesteps'] = self.timesteps
        
        
    def timestep_GBM(self, sim_num: int):
        #each timestep is the overall time divided by how many we're checking
        timestep = self.time / self.timesteps
        
        paths = np.zeros((sim_num, self.timesteps + 1))
        paths[:, 0] = self.init_price
        #A numpy array of a bunch of normalized random numbers, with the 2d shape of the sim_num x timesteps
        Z = np.random.normal(0, 1, (sim_num, self.timesteps))
        

        #Each part of the equation for GBM, divided into chunks
        drift = timestep * (self.rf_rate - (0.5 * self.vol ** 2))
        diff = self.vol * np.sqrt(timestep)
        
        #This returns another numpy array
        returns = drift + diff * Z
        
        for t in range(1, self.timesteps + 1):
           paths[:, t] = paths[:, t - 1] * np.exp(drift + diff * Z[:, t-1])
        
        return paths
    
    def american_put(self, sim_num: int):
        
        #Making a list of paths with timesteps using our last function
        paths = self.timestep_GBM(sim_num)
        timestep = self.time / self.timesteps
        
        #the discount equation to include the risk-free rate accurately
        disc = np.exp( -1 * self.rf_rate * timestep)
        
        #all the possible payoffs at any given point, since you cant lose money on an option
        payoffs = np.maximum(self.strike_price - paths, 0)
        
        #cash flows is just our going backwards through the payoffs
        flows = payoffs[:, -1]
        
        for t in range(self.timesteps + 1):
            
            #All parts of the payoff that are green
            profits = payoffs[:, t]>0
            
            #These are used for the fitting least squares regression
            #This allows us to predict the possibility of better future cashouts
            regress_x = paths[profits, t][np.newaxis].T
            regress_y = flows[profits] * disc
            
            #checking the length is not 0
            if not len(regress_x):
                continue
            
            #just horizontally stacking the polynomial functions
            #I choose poly degree of 2 bcuz this is running on my laptop, not a huge database
            funcs = np.hstack([np.ones_like(regress_x), regress_x, regress_x ** 2])
            
            #Least-squares algorithm regression fitting to get coefficients
            coeffs = np.linalg.lstsq(funcs, regress_y)[0]
            
            #Multiplying the matrices to get continuation estimate
            cont = np.matmul(funcs, coeffs)
        
            #checking if we're above the continuation estimate, and updating
            put_now = payoffs[profits, t] > cont
            update = np.where(profits)[0][put_now]
            
            flows[update] = payoffs[update, t]
            
            #Discounting
            flows *= disc
            
        return np.mean(flows)
    
    def american_call(self, sim_num):
        paths = self.timestep_GBM(sim_num)
        timestep = self.time / self.timesteps
        disc = np.exp( -1 * self.rf_rate * timestep)
        
        #This is literally the only different step from put
        payoffs = np.maximum(paths - self.strike_price, 0)
        
        flows = payoffs[:, -1]
        
        for t in range(self.timesteps + 1):
            
            profits = payoffs[:, t] > 0
            
            regress_x = paths[profits, t]
            regress_y = flows[profits] * disc
            
            if not len(regress_x):
                continue
            
            funcs = np.vstack([np.ones_like(regress_x), regress_x, regress_x ** 2])
            coeffs = np.linalg.lstsq(funcs.T, regress_y)[0]
            cont = np.matmul(funcs.T, coeffs)
        
            put_now = payoffs[profits, t] > cont
            update = np.where(profits)[0][put_now]
            
            flows[update] = payoffs[update, t]
            
            flows *= disc
            
        return np.mean(flows)

    def delta(self, bump_percentage: float, sim_num: int) -> float:
        #The delta of an option shows how much it changes as the underlying asset changes
        #There are closed formulas for this, but simulating with MC is more in line with this portfolio
        
        #The og_price is just the current option call price
        og_price = self.american_call(sim_num)
        
        #This calculates the call price if we bumped the initial price by a tiny bit
        bump_option = AmericanOptions(self.strike_price, self.vol, self.rf_rate, self.time, self.init_price * (1 + bump_percentage), self.timesteps)
        bumped_price = bump_option.american_call(sim_num)
        
        #Finding the output difference over the bump difference
        delta = (bumped_price - og_price) / (self.init_price * bump_percentage)
        return delta
    
    

american = AmericanOptions(100, 0.2, 0.04, 1, 100, 20)
print(american.american_call(1000))
