#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:10:55 2018

PLOT THE FT FUNCTION FOR DIFFERENT TAU AND SIGMA VALUES
@author: sophie
"""

import define_functions as f
import matplotlib.pyplot as plt
import numpy as np

# Initialise the function object and import the data
func = f.DecayFunction()
ts, sigmas = func.import_data()

# Creating lists of tau and sigma values, 
# changing both and changing only one at a time

tau_range = [np.linspace(0.1,0.5, 4), 
             np.linspace(0.1,0.5, 4), 
             np.linspace(0.1,0.1, 4)]

sigma_range = [np.linspace(0.1, 0.5, 4), 
               np.linspace(0.1,0.1, 4), 
               np.linspace(0.1,0.5, 4)]

captions = ['F$^m$(t) for different $\sigma$ and $\\tau$', 
            'F$^m$(t) for different $\\tau$', 
            'F$^m$(t) for different $\sigma$']
i = 0

# Plotting each graph to observe how tau and sigma affect the function f^m(t)
for t_range, s_range in zip(tau_range, sigma_range):
    
    print('\n', captions[i])
    # Creating a new figure for each of the three scenarios
    plt.figure()
    
    for t,s in zip(t_range, s_range):
    
        fm = func.fm_function(ts, s, t)
        area_simps = f.integrate_simpson(ts.min(), 
                                         ts.max(), func.fm_function, 
                                         s, t, n=len(ts))
        
        print('tau= %.2f, sigma= %.2f, area= ' %(t, s), area_simps)
        
        plt.plot(ts, fm, 
                 label='$\sigma$= %.2f, $\\tau$=%.2f'%(s, t))
        plt.xlabel('time ($p$s)', fontsize=15)
        plt.ylabel('f$^m$(t)', fontsize=15)
        plt.title(captions[i])
        plt.legend(prop={'size': 14})
        plt.grid(b=True)
        plt.show()
    
    # Change the caption to match each figure
    i+=1
