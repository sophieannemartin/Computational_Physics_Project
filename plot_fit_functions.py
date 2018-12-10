#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:10:55 2018

PLOT THE FT FUNCTION FOR DIFFERENT TAU AND MU VALUES
@author: sophie
"""

import define_functions as f
import matplotlib.pyplot as plt
import numpy as np

func = f.DecayFunction()
ts, sigmas = func.import_data()

tau_range = np.linspace(0.1,0.5, 4)
sigma_range = np.linspace(0.1, 0.5, 4)

for t, s in zip(tau_range, sigma_range):
    
    fm = func.fm_function(ts, s, t)
    area = np.trapz(fm, ts)
    print(area)
    
    plt.plot(ts, fm, 
             label='$\sigma$= %.2f, tau=%.2f'%(s, t))
    plt.xlabel('time ($p$s)')
    plt.ylabel('f$_m$(t)')
    plt.title('F$_m$(t) for different $\sigma$ and tau, A=800')
    plt.legend()
    plt.grid(b=True)
    plt.show()