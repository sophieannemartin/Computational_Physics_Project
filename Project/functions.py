"""
DEFINING FUNCTIONS TO MANIPULATE DATA
21/11/18
SOPHIE MARTIN
"""# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.special import erfc

def import_data():
    file = '/Users/sophie/Documents/Work/Year 3/Courses/Computational Physics/Project/lifetime-2018.csv'
    # Import data as dataframe with the two columns t and sigma
    data = pd.read_csv(file, header=None).rename(columns={0: 't', 1: 'sigma'}).sort_values(by=['t'])
    return data


# Define gaussian function to be used to fit to the data
def gauss(x, *p):
    mu, sigma = p
    return np.exp(-(x-mu)**2/(2*sigma**2))


def fm_function(t, sigma, tau):
    f_m = ((1/(2*tau))*np.exp(((sigma**2)/(2*tau**2))-(t/tau))*
           erfc((1/np.sqrt(2))*((sigma/tau) - (t/sigma))))
    return f_m


def cosh(x):
    coshx = np.cosh(x)
    return coshx


def find_NLL_value(function, ts, sigmas, tau):
    
    # Initialise nll summation 
    nll = 0
    n = len(ts)
    
    for i in range(n): #0 to n-1 instead of 1 to nprint(i)
        nll += np.log10(fm_function(ts[i], sigmas[i], tau))
    return -nll

        
def plot_NLL(initial_u, dataframe): #initial_u = range of tau values
    
    # Define fixed paramters
    t = dataframe['t'].values
    sigma = dataframe['sigma'].values
    
    nll_list = []
    
    for u in initial_u:
    
        nll = find_NLL_value(t, u, sigma)
        nll_list.append(nll)
    
    return nll_list, initial_u


def find_parabolic_x3(values, ts, sigmas):
    
    numerator = (((values[2]**2 - values[1]**2)*
              find_NLL_value(ts, sigmas, values[0])) + 
              ((values[0]**2 - values[2]**2)*
              find_NLL_value(ts, sigmas, values[1])) +
              ((values[1]**2 - values[0]**2)*
              find_NLL_value(ts, sigmas, values[2])))
              
    denominator = (((values[2] - values[1])*
          find_NLL_value(ts, sigmas, values[0])) + 
          ((values[0] - values[2])*
          find_NLL_value(ts, sigmas, values[1])) +
          ((values[1] - values[0])*
          find_NLL_value(ts, sigmas, values[2])))
              
    x3 = 0.5*(numerator/denominator)
    return x3

    
def remove_highest(values, ts, sigmas): # from 4 values
    
    nll_values = []
    
    for each in values:
        nll_val = find_NLL_value(ts, sigmas, each)
        nll_values.append(nll_val)
        
    index = np.argmax(nll_values)
    values.pop(index)
    return values


def find_initial_values(function_values, test_values):
    
    """
    Finds the initial guesses by finding the minimum in the test array
    Therefore limited by the spacing used to plot the NLL so will not be correct minimum
    """
    
    min_index = np.argmax(function_values)
    guess = test_values[min_index]
    adj = test_values[min_index+1]
    prev = test_values[min_index-1]
    initial_values = [prev, guess, adj]
    
    return initial_values


def parabolic_minimiser(data, initial_guess, threshold):
    
    """
    Parabolic 1D minimiser finds the minimum using the negative log likelihood
    Stops dependant on a user-defined threshold difference
    """
    ts = data['t'].values
    sigmas = data['sigma'].values
    
    values = initial_guess
    # Initialise a list to keep track of x3 in order to calculate difference
    x3_list = [0.4]
    difference = 1.0
    iterations = 0
    
    while difference > threshold:
        x3 = find_parabolic_x3(values, ts, sigmas)
        x3_list.append(x3)
        difference = np.abs(x3-x3_list[-2])
        values.append(x3)
        values = remove_highest(values, ts, sigmas)
        iterations+= 1
      
    print('Minimum Found!: ', x3)
    return x3, iterations, x3_list
    