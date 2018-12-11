#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:54:05 2018

TESTING 1D MINIMISER ON THE COSHX FUNCTION
@author: sophie
"""

import numpy as np
import matplotlib.pyplot as plt
import minimiser

# Define coshx function to be used to test the minimisation function
def cosh(x):
    coshx = np.cosh(x)
    return coshx


def main():
    
    x_range = np.linspace(-10,10, 100)
    y_values = cosh(x_range)
    
    # Run minimisation function tp obtain value, number of iterations and
    # list of x3 values that were obtained in the algorithm
    
    minimum, iterations, x3_list = minimiser.minimise_1D(
                x_range, cosh, np.finfo(float).eps)
    
    
    x3_mins = []
        
    for x3 in x3_list:
        val = cosh(x3)
        x3_mins.append(val)
            
            
    plt.plot(x_range, y_values)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Minimisation of cosh(x)')
    plt.grid()
    plt.plot(x3_list, x3_mins, '.', color='red')
    plt.show()
    
if __name__ == "__main__":
    main()