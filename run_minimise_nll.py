#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:35:07 2018

PLOTTING THE NLL FIT AS A FUNCTION OF TAU

@author: sophie
"""

import define_functions as f
import matplotlib.pyplot as plt
import numpy as np
import minimiser

def main():
    
    # Define a range of tau values to apply the minimiser over
    # Assumes that there is a minimum within this range
    # Will find local minimum in range
    
    taus_range = np.linspace(0.1, 4.0, 100)
     
    decayfunction = f.DecayFunction()
    nll_values = decayfunction.get_nll_values(taus_range)
    
    minimum, iterations, x3_list = minimiser.minimise_1D(
            taus_range, decayfunction.get_nll_values, np.finfo(float).eps)
    
    nll_mins = []
    
    for tau in x3_list:
        val = decayfunction.find_nll_value(tau)
        nll_mins.append(val)
        
    plt.plot(taus_range, nll_values)
    plt.plot(x3_list, nll_mins, '.', color='red')
    plt.xlabel('tau')
    plt.ylabel('NLL value')
    plt.grid()
    plt.title('NLL(tau) for different tau')
    plt.show()

if __name__ == "__main__":
    main()