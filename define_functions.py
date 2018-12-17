"""
DEFINING FUNCTIONS
21/11/18
SOPHIE MARTIN
"""

import pandas as pd
import numpy as np
from scipy.special import erfc
import sys

# ----------------------------------

# Define gaussian function to be used to fit to the data

def gauss(x, *p):
    mu, sigma = p
    return np.exp(-(x-mu)**2/(2*sigma**2))

# Define integration method
    
def integrate_simpson(a, b, function, *params, n=500):
    
    dx = (b-a)/n
    x = np.linspace(a, b, n+1)
    y = function(x, *params)
    integral = 0
    
    integral = np.sum(y[:-1:2] + 4*y[1::2] + y[2::2])
    
    val  = (dx/3) * integral
    return val
    
# ----------------------------------

class DecayFunction:
    
    """
    DecayFunction Object
    Class that has all the relevant function specific to the t, sigma
    dataset such as computing the NLLs, and fm, signal and background functions.
    
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
        return t, sigma # Return arrays 

    def fm_function(self, t, sigma, tau):
        # Input parameters can be single-valued or arrays
        f_m = ((1/(2*tau))*np.exp(((sigma**2)/(2*tau**2))-(t/tau))*
               erfc((1/np.sqrt(2))*((sigma/tau) - (t/sigma))))
        return f_m
    

    def fm_background(self, t, sigma):
        # Defines background function
        bckg = ((1/(sigma*np.sqrt(2*np.pi)))*
                (np.exp(-0.5*((t**2)/(sigma**2)))))
        return bckg
    
    
    def signal_and_background(self, t, sigma, tau, a):
        # Defines the signal & background value
        val  = (a*(self.fm_function(t, sigma, tau))+
                ((1-a)*(self.fm_background(t, sigma))))
        return val
    
    
    def get_data(self):
        return self.__t__, self.__sigma__


    def change_n(self, n):
        self.__n__ = n
        
# ------------------------------------
    # Code for the NLL based on one parameter estimate

    def find_nll_value(self, u):
        
        # Initialise nll summation 
        nll = 0
        
        for i in range(self.__n__): # 0 to n-1 instead of 1 to n
            nll += np.log(self.fm_function(self.__t__[i], self.__sigma__[i], u))
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
    
# ------------------------------------
        
    # Defining the nll for two parameters u1 and u2 estimates
    
    def get_2d_nll_values(self, u1_range, u2_range): 
        
        # u_range = range of tau values input
        
        if type(u1_range) == list and type(u2_range) == list:
            
            nll_values = []
            for u1, u2 in [(x,y) for x in u1_range for y in u2_range]:
                nll = self.find_2d_nll_value(u1, u2)
                nll_values.append(nll)
            
        else:
            nll_values = self.find_2d_nll_value(u1_range, u2_range)
    
        return nll_values
    
    
    def find_2d_nll_value(self, u1, u2):
        
        # A cannot be larger than 1
        if u2 > 1:
            sys.exit('invalid a')
            
        # Initialise nll summation 
        nll = 0
        
        for i in range(self.__n__): # 0 to n-1 instead of 1 to n
            nll += np.log(self.signal_and_background(self.__t__[i], 
                                                      self.__sigma__[i], u1, u2))
        return -nll


# ----------------------------------
