"""
DEFINING FUNCTIONS
21/11/18
SOPHIE MARTIN
"""

import pandas as pd
import numpy as np
from scipy.special import erfc

# ----------------------------------

# Define gaussian function to be used to fit to the data
def gauss(x, *p):
    mu, sigma = p
    return np.exp(-(x-mu)**2/(2*sigma**2))

# Define coshx function to be used to test the minimisation function
def cosh(x):
    coshx = np.cosh(x)
    return coshx

# ----------------------------------

class DecayFunction:
    
    """
    DecayFunction Object
    Class that has all the relevant function specific to the t, sigma
    dataset such as computing the NLL, and fm, signal and background functions.
    
    """

    def __init__(self):
        self.__t__, self.__sigma__ = self.import_data()
        self.__n__ = len(self.__t__)
        
    
    def import_data(self):
        file = '/Users/sophie/Documents/Work/GitHub/Year 3/Computational_Physics_Project/lifetime-2018.csv'
        # Import data as dataframe with the two columns t and sigma sorted by ascending t
        data = pd.read_csv(file, header=None).rename(columns={0: 't', 1: 'sigma'}).sort_values(by=['t'])
        t = data['t'].values
        sigma = data['sigma'].values
        return t, sigma # These are arrays of the values
    
    
    def get_data(self):
        return self.__t__, self.__sigma__
    
    
    def find_nll_value(self, u):
        
        # Initialise nll summation 
        nll = 0
        
        for i in range(self.__n__): # 0 to n-1 instead of 1 to n
            nll += np.log10(self.fm_function(self.__t__[i], self.__sigma__[i], u))
        return -nll

    
    def get_nll_values(self, u_range): # u_range = range of tau values
        
        
        if type(u_range) == list:
            nll_values = []
            for u in u_range:
                nll = self.find_nll_value(u)
                nll_values.append(nll)
        
        else:
            nll_values = self.find_nll_value(u_range)
    
        return nll_values
    
    # Careful with the use of t, sigma and tau here
    # Can define them as all t and tau values specified in the function itself
    
    def fm_function(self, t, sigma, tau):
        f_m = ((1/(2*tau))*np.exp(((sigma**2)/(2*tau**2))-(t/tau))*
               erfc((1/np.sqrt(2))*((sigma/tau) - (t/sigma))))
        return f_m
    

    def fm_background(self):
        
        bckg = ((1/(self.__sigma__**np.sqrt(2*np.pi)))*
                np.exp(-0.5*(self.__t__**2/self.__sigma__**2)))
        return bckg
    
    
    def signal_and_background(self, a, tau):
        val  = (a*(self.fm_function(tau))+
                (1-a)*(self.fm_background()))
        return val
        

# ----------------------------------
