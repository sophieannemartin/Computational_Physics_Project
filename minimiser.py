#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:19:53 2018

@author: sophie
"""

import numpy as np

# 1D Parabolic Minimiser Definition

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


def minimise_1D(x_values, function, maxiter):
    
    """
    Parabolic 1D minimiser finds the minimum using the negative log likelihood
    Stops dependant on a user-defined iteration limit 
    Will stop before this if the difference is less than machine accuracy
    """
    
    fvalues = function(x_values)
    
    values = find_initial_values(x_values, fvalues)
    # Initialise a list to keep track of x3 in order to calculate difference
    x3_list = [] # Initialise empty list  
    difference = np.finfo(float).eps
    iterations = 0
    
    while difference > np.finfo(float).eps or iterations<maxiter:
        x3 = find_parabolic_x3(values, function)
        x3_list.append(x3)
        
        if len(x3_list) > 1: 
            # Once you have enough values of x3, start calculating difference
            difference = np.abs(x3-x3_list[-1])
            
        values.append(x3)
        values = remove_highest(values, function)
        iterations+= 1
      
    print('Minimum Found!: ', x3)
    return x3, iterations, x3_list


# --------------------- 
    
# Quasi-Newton 2D Minimiser Definition
    
def central_difference(param1, param2, function, h):
    
    # Initialise gradf with 2 rows and one column
    grad_f = np.zeros((2,1))
    
    #df/dx at constant y
    grad_f[0,0] = (function(param1+h, param2)-function(param1-h, param2))/2*h
    #df/dy at constant x
    grad_f[1,0] = (function(param1, param2+h)-function(param1, param2-h))/2*h

    return grad_f


def dfp_update(g_list, x_list, gradf_list):
    
    delta_vector = x_list[-1]-x_list[-2]
    gamma_vector = gradf_list[-1]-gradf_list[-2]
    
    outer = np.outer(delta_vector, delta_vector)
    
    g = g_list[-1] + (outer/
              (np.dot(np.transpose(gamma_vector), delta_vector))) - \
              (np.dot(outer, (np.dot(g_list[-1], outer)))/
               (np.dot(np.transpose(gamma_vector), (np.dot(g_list[-1],gamma_vector)))))
    return g

def find_initial_vectorx(u1_range, u2_range, function):
    
        # Automatic selection of best initial position based on a range of values
        X, Y = np.meshgrid(u1_range, u2_range)
        zs = np.array([function(x,y) for x,y in zip(np.ravel(u1_range), np.ravel(u2_range))]) 
        
        for u1, u2 in zip(u1_range, u2_range):
            z = function(u1, u2)
            
            if z == np.amin(zs):
                initialu1, initialu2 = u1,u2
                
        return initialu1, initialu2, zs
    
    
def minimise_quasi_newton(x0, y0, function, h, maxiter, alpha):
    
    """
    x0 and y0 are used to define the initial stating position
    alpha must be < 0.1 to work.. (why?????)
    
    x0 and y0 are initial position therefore need a starting point
    """
    
    g_list  = []
    x_list = []
    gradf_list = []
    
    # Create G0 and x0 to begin
    gn = np.zeros((2,2))
    for i in range(gn.shape[0]):
        gn[i,i] = 1
        
    vector_x =  np.array([[x0], [y0]])

    gradf = central_difference(vector_x[0,0], vector_x[1,0], function, h)
    
    g_list.append(gn)
    x_list.append(vector_x)
    gradf_list.append(gradf)
    
    difference = np.array([[threshold], [threshold]])
    
    while difference.all() > 0:
        
        print(vector_x)
        # Gradient at a specific x value
        vector_x = x_list[-1] - np.dot(alpha*g_list[-1], gradf_list[-1]) # Use last G and f
        # Subtraction of two vectors
        difference = (np.around(vector_x, 12))-(np.around(x_list[-1],12)) 
        print(difference)
        # If not minimum add to the list
        x_list.append(vector_x)
        gradf = central_difference(vector_x[0,0], vector_x[1,0], function, h)
        # Update gradf (at new x, y)
        gradf_list.append(gradf)
        newg = dfp_update(g_list, x_list, gradf_list)
        
        # TO DO - When same pair of values is obtained - define a quit regime (
        # i.e. zero delta vector)
        
        # Update G (Hessian) using new x,y, and new gradf
        g_list.append(newg)
        # Repeats this until minimum is found

    print('Minimum Found!: ', vector_x)
    return vector_x, x_list
