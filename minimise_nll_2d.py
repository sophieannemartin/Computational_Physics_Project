#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 18:28:12 2018

@author: sophie
"""

import define_functions as f
import matplotlib.pyplot as plt
import numpy as np
import minimiser
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
    

def main():
    
    # Define a range of tau values and alpha to apply the minimiser over
    # Assumes that there is a minimum within this range
    # Will find local minimum in range
    
    decayfunction = f.DecayFunction()

    # Useful to find the initial guess
    taus_range = np.linspace(0.4, 0.5, 20)
    
    # alpha cannot be > 1
    alpha_range = np.linspace(0.9, 1, 20)
    
    X, Y = np.meshgrid(taus_range, alpha_range)
    initialtau, initialalpha, zs = minimiser.find_initial_vectorx(taus_range, alpha_range,
                                                                  decayfunction.get_2d_nll_values)
    Z = zs.reshape(X.shape)

    minimum, min_list, iterations = minimiser.minimise_quasi_newton(
            initialtau, initialalpha, decayfunction.find_2d_nll_value, 
            0.00001, 0.000001, maxiter=500)
    

    # Plotting nll over different tau and alpha to decide on best minimum point
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.set_xlabel('tau')
    ax.set_ylabel('alpha')
    ax.set_zlabel('NLL value')
    ax.set_title('Plotting the NLL over different alpha and tau')

    ax.plot([minimum[0,0]],[minimum[1,0]], [decayfunction.find_2d_nll_value(minimum[0,0], minimum[1,0])],
        markerfacecolor='yellow', markeredgecolor='yellow', marker='o', 
        markersize=3, alpha=1)
        
    plt.show()

    
    return minimum, min_list, iterations

if __name__ == "__main__":
    minimum, min_list, iterations = main()