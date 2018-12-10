#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 17:00:27 2018

@author: sophie
"""

import pandas as pd
import numpy as np
from scipy.special import erfc

def import_data():
    file = '/Users/sophie/Documents/Work/GitHub/Year 3/Computational_Physics_Project/lifetime-2018.csv'
    # Import data as dataframe with the two columns t and sigma sorted by ascending t
    data = pd.read_csv(file, header=None).rename(columns={0: 't', 1: 'sigma'}).sort_values(by=['t'])
    t = data['t'].values
    sigma = data['sigma'].values
    return t, sigma # These are arrays of the values


def fm_function(t, sigma, tau):
    f_m = ((1/(2*tau))*np.exp(((sigma**2)/(2*tau**2))-(t/tau))*
           erfc((1/np.sqrt(2))*((sigma/tau) - (t/sigma))))
    return f_m


def cosh(x):
    coshx = np.cosh(x)
    return coshx


# ---------------    

def find_NLL_value(self, u=None):
        
        # Initialise nll summation 
        nll = 0
        n = len(self.__args__[0])
        
        for i in range(n): # 0 to n-1 instead of 1 to n
            nll += np.log10(self.__function__(self.__fixed_args__, u))
        return -nll

    
