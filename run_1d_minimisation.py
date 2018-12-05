#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:35:07 2018

PLOTTING THE NLL FIT AS A FUNCTION OF TAU

@author: sophie
"""

import functions as funcs
import matplotlib.pyplot as plt
import numpy as np

data = funcs.import_data()

taus_range = np.linspace(0.1, 4, 100)
nlls, taus = funcs.plot_NLL(taus_range, data)

initial_taus = funcs.find_initial_values(nlls, taus)

minimum, iterations, x3_list = funcs.parabolic_1d_minimiser(data, initial_taus, np.finfo(float).eps)
nll_mins = []

for tau in x3_list:
    val = funcs.find_NLL_value(data['t'].values, data['sigma'].values, tau)
    nll_mins.append(val)
    
plt.plot(taus, nlls)
plt.plot(x3_list, nll_mins, '.', color='red')
plt.xlabel('tau')
plt.ylabel('NLL value')
plt.grid()
plt.title('NLL(tau) for different tau')
plt.show()
