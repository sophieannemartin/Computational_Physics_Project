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
            taus_range, decayfunction.get_nll_values, maxiter=1000)
    
    nll_mins = []
    
    for tau in x3_list:
        val = decayfunction.find_nll_value(tau)
        nll_mins.append(val)
    
    print('Calculating standard deviation using interpolation...')
    root1, root2 = minimiser.find_standard_deviation(minimum, taus_range, 
                                                         decayfunction.get_nll_values)
    
    gauss_sigma = minimiser.gauss_standard_deviation(x3_list, 
                                                    decayfunction.get_nll_values)
    
    print('tau-: %f, tau+: %f' % (root1, root2))
    print('sigma-: %f, sigma+: %f' % (minimum-root1, root2-minimum))
    print('Gauss method sigma: ', gauss_sigma)
    
    plt.figure(1)
    plt.plot(taus_range, nll_values)
    plt.plot(x3_list[:-1], nll_mins[:-1], '.', color='red', label='Minimum iterations')
    plt.plot(x3_list[-1], nll_mins[-1], '.', color='green', label='Minimum found')
    plt.xlabel('tau')
    plt.ylabel('NLL value')
    plt.grid()
    plt.title('NLL(tau)')
    plt.legend()
    
    plt.figure(2)
    plt.plot(taus_range, nll_values)
    plt.xlabel('tau')
    plt.ylabel('NLL value')
    plt.grid()
    plt.title('NLL(tau)')
    
    smaller_range = np.linspace(0.2, 0.8, 1000)
    nll_2 = decayfunction.get_nll_values(smaller_range)
    
    plt.figure(3)
    plt.plot(smaller_range, nll_2)
    plt.plot(x3_list[:-1], nll_mins[:-1], '.', color='red', label='Minimum iterations')
    plt.plot(x3_list[-2], nll_mins[-2], '.', color='green', label='Minimum found')
    plt.xlabel('tau')
    plt.ylabel('NLL value')
    plt.grid()
    plt.title('NLL(tau) near minimum')
    plt.legend()
    plt.xlim(0.395,0.42)
    plt.ylim(2695, 2712)
    plt.show()

    return minimum, iterations, x3_list

if __name__ == "__main__":
    minimum, iterations, x3_list = main()