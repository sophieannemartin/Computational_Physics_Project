#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:54:05 2018

TESTING 1D MINIMISER ON THE COSHX FUNCTION
@author: sophie
"""

import define_functions as f
import numpy as np
import matplotlib.pyplot as plt
import minimiser

def main():
    
    x_range = np.linspace(-10,10, 100)
    y_values = f.cosh(x_range)
    minimum, iterations, x3_list = minimiser.minimise_1D(
                x_range, f.cosh, np.finfo(float).eps)
    
    
    x3_mins = []
        
    for x3 in x3_list:
        val = f.cosh(x3)
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