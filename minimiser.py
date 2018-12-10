#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:19:53 2018

@author: sophie
"""

import numpy as np


"""

Minimisation functions with corresponding functions to calculate a minimum value
from a set of x and corresponding yvalues but using a 1D parabolic 
minimiser algorithm.

"""
    
def find_parabolic_x3(values, function):
    
    numerator = (((values[2]**2 - values[1]**2)*
              function(values[0])) + 
              ((values[0]**2 - values[2]**2)*
              function(values[1])) +
              ((values[1]**2 - values[0]**2)*
              function(values[2])))
              
    denominator = (((values[2] - values[1])*
          function(values[0])) + 
          ((values[0] - values[2])*
          function(values[1])) +
          ((values[1] - values[0])*
          function(values[2])))
              
    x3 = 0.5*(numerator/denominator)
    return x3

    
def remove_highest(values, function): # from 4 values
        
    index = np.argmax(function(values))
    values.pop(index)
    return values


def find_initial_values(test_values, fvalues):
    
    """
    Finds the initial values by finding the minimum in the test array
    Limited by the spacing used to plot the NLL so will not be correct minimum
    """
    
    min_index = np.argmin(fvalues)
    guess = test_values[min_index]
    adj = test_values[min_index+1]
    prev = test_values[min_index-1]
    initial_values = [prev, guess, adj]
    
    return initial_values


def minimise_1D(x_values, function, threshold):
    
    """
    Parabolic 1D minimiser finds the minimum using the negative log likelihood
    Stops dependant on a user-defined threshold difference
    """
    
    fvalues = function(x_values)
    
    values = find_initial_values(x_values, fvalues)
    # Initialise a list to keep track of x3 in order to calculate difference
    x3_list = [0.4]
    difference = 1.0
    iterations = 0
    
    while difference > threshold:
        x3 = find_parabolic_x3(values, function)
        x3_list.append(x3)
        difference = np.abs(x3-x3_list[-2])
        values.append(x3)
        values = remove_highest(values, function)
        iterations+= 1
      
    print('Minimum Found!: ', x3)
    return x3, iterations, x3_list