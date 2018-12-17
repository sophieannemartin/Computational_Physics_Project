"""
DEFINING THE PARABOLIC MINIMSER
10/12/18
@author: SOPHIE MARTIN
"""

import numpy as np

def find_parabolic_x3(values, function):
    
    """
    Function that uses the parabolic approximation to estimate the minimum value
    
    Inputs: values - a list containing 3 values (x)
    function - the function used to calculate the corresponding f(x)
    
    Returns: An estimated minimum
    """
    
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
    
    Inputs: test_values - a list/array of values to check through
    fvalues - the corresponding function values at each test value
    
    Returns: An list of 3 values which can be fed into the minimiser
    """
    
    min_index = np.argmin(fvalues)
    guess = test_values[min_index]
    adj = test_values[min_index+1]
    prev = test_values[min_index-1]
    initial_values = [prev, guess, adj]
    
    return initial_values


def minimise_1D(x_values, function, maxiter=500):
    
    """
    Parabolic 1D minimiser finds the minimum using the negative log likelihood
    Stops dependant on a user-defined max number of iterations
    Will stop before this is difference is less than the threshold value
    
    Inputs: x_values - a range of x_values
    function - the function to minimise
    maxiter = the maximum number of iterations to perform before exiting
    """
    
    fvalues = function(x_values)
    
    values = find_initial_values(x_values, fvalues)
    # Initialise a list to keep track of x3 in order to calculate difference
    x3_list = [] # Initialise empty list
    iterations = 0
    repeat= True
    
    while repeat==True and iterations<maxiter:
        
        # Find x3 value from the 3 initial values
        x3 = find_parabolic_x3(values, function)
        
        # Check if the same as before
        if len(x3_list) > 0 and x3==x3_list[-1]:
            x3 = x3_list[-1]
            repeat = False
            print('Minimum Found!: ', x3)
        
        # Also check the nll value is still decreasing
        elif len(x3_list) > 0 and function(x3) > function(x3_list[-1]):
            x3 = x3_list[-1]
            repeat = False
            print('Minimum Found! ', x3)
        
        # If none of the above is true,
        # Store current x3 and remove highest value out of the 4 values
        # Repeat to find a new x3
        else:
            x3_list.append(x3)
            values.append(x3)
            values = remove_highest(values, function)
            iterations+= 1
        
    if iterations >= maxiter:
        print('Maximum number of iterations reached before convergence!')
        print('Best Estimate of Minimum: ', x3)
     
    return x3, iterations, x3_list


def secant_root_finder(interval, function, shift=0):
    
    """
    Root finder that uses the secant method.
    Requires initial interval that must contain ONE value at which function = 0
    If not, then breaks.
    
    Inputs: interval - a list containing the edges of the interval containing a root
    function - the function to solve for.
    
    Can shift the initial function down by a fixed value where required 
    to find the value of x at any f(x)-shift value with the shift parameter.
    """
    
    a, b = interval[0], interval[1]
    
    if (function(a)-shift)*(function(b)-shift) >= 0:
        print("Error! Incorrect interval - Secant method will fail.")
        return None
    
    while np.abs(b-a) > 0.0000000000000001:
        
        x_m = a - (function(a)-shift)*((b-a)/((function(b)-shift)
        -(function(a)-shift)))
        f_x_m = function(x_m)-shift
        
        if (function(a)-shift)*f_x_m < 0:
            a = a
            b = x_m
            
        elif (function(b)-shift)*f_x_m < 0:
            a = x_m
            b = b
            
        elif f_x_m == 0:
            # Exact solution found
            print('Found exact solution!')
            return x_m
        
        else:
            print('Secant method fails')
            return None  
    

def find_standard_deviation(minimum_pos, function):
    
    """
    Calculates the standard deviation in the positive and negative direction 
    where the function changes by 0.5
    
    Inputs: minimum_pos - the position of the minimum
    function - the function being applied to
    """
    
    shift = function(minimum_pos) + 0.5
    
    # Define initial interval using 0.1 either side of the minimum
    # Can see from the plot that this is sufficient in this specific case
    
    starting_interval = [0.3, 0.4]
    minus = secant_root_finder(starting_interval, function, shift)
        
    starting_interval = [0.4, 0.5]
    plus = secant_root_finder(starting_interval, function, shift)
    return minus, plus


def gauss_standard_deviation(x_list, function):
    
    """
    Uses the minimum and 2 prior values to calculate an error on the minimum.
    Calculates an error based on their curvatures by fitting a P2(x)
    Inputs: x_list - the list of minimum found
    function - the function being applied to
    """
    
    x0 = x_list[-3]
    x1 = x_list[-2]
    x2 = x_list[-1]
    
    # Calculate the required denominators
    a0 = 1/((x0-x1)*(x0-x2))
    a1 = 1/((x1-x0)*(x1-x2))
    a2 = 1/((x2-x0)*(x2-x1))
    
    # Alpha is given by the co-efficient of the 2nd order lagrange polynomial
    alpha = a0*function(x0) + a1*function(x1) + a2*function(x2)
    sigma = np.sqrt(1/(2*alpha))
    return sigma
    
# --------------------- 
    
# Quasi-Newton 2D Minimiser Definition
    
def central_difference(param1, param2, function, h):
    
    """
    Uses the central difference scheme to approximate the gradient to order
    h^2 by calculating partial derivatives for a multi-variate function of
    two variables.
    
    Inputs: param1, param2 - the paramters of the function
    function - the function to differentiate
    h - the step size
    
    Returns: A 2 x 1 vector containing the two partial derivatives
    """
    
    # Initialise gradf with 2 rows and one column
    grad_f = np.zeros((2,1))
    
    #Partial df/dx at constant y
    grad_f[0,0] = (function(param1+h, param2)-function(param1-h, param2))/2*h
    #Partial df/dy at constant x
    grad_f[1,0] = (function(param1, param2+h)-function(param1, param2-h))/2*h

    return grad_f


def dfp_update(g_list, x_list, gradf_list):
    
    """
    Function to implement the Davidon-Fletcher-Powell method to update
    the estimation of the Hessian as part of the quasi-Newton minimum search
    """
    
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
        zs = np.array([function(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
        
        count = 0
        for u1, u2 in zip(np.ravel(X), np.ravel(Y)):
            
            if count == np.argmin(zs):
                initialu1, initialu2 = u1,u2
            
            count += 1
                
        return initialu1, initialu2, zs
    
    
def minimise_quasi_newton(x0, y0, function, h, alpha, maxiter=100):
    
    """
    x0 and y0 are used to define the initial stating position
    alpha must be < 0.1 to work
    
    x0 and y0 are initial positions that must be specified
    
    Stops based on maximum iterations to avoid blow ups
    """
    
    g_list  = []
    x_list = []
    gradf_list = []
    
    # Create G0 and x0 to begin
    gn = np.array([[1,0], [0,1]])
    vector_x =  np.array([[x0], [y0]])

    gradf = central_difference(vector_x[0,0], vector_x[1,0], function, h)
    
    g_list.append(gn)
    x_list.append(vector_x)
    gradf_list.append(gradf)
    
    difference = np.array([[0.00002], [0.00002]])
    repeat = True
    iterations = 0
    
    print('Minimisation start..')
    
    while repeat == True and iterations<maxiter:
        
        vector_x = x_list[-1] - np.dot(alpha*g_list[-1], gradf_list[-1]) # Use last G and f
        
        if vector_x[0,0] == (x_list[-1])[0,0] and vector_x[1,0] == (x_list[-1])[1,0]:
            repeat = False
        
        # If not minimum add to the list to keep track
        else:
            x_list.append(vector_x)
            difference = vector_x - x_list[-2]
            gradf = central_difference(vector_x[0,0], vector_x[1,0], function, h)
            # Update gradf (at new x, y)
            gradf_list.append(gradf)
            newg = dfp_update(g_list, x_list, gradf_list)
            
            # Update G (Hessian) using new x,y, and new gradf
            g_list.append(newg)
            iterations+=1
            # Repeats this until minimum is found or max iterations are reached
            
    if iterations >= maxiter:
        print('Maximum number of iterations reached before convergence! Deltas between last values:[%f, %f]'
          % (difference[0,0], difference[1,0]))
        print('Best Minimum Estimate: ', vector_x)
            
    else:
        print('Exact Minimum Found!: ', vector_x)

    return vector_x, x_list, iterations


def find_covariance_error(param1, param2, function, h):
    
    matrix = np.zeros((2,2))
    matrix[0,0] = (function(param1+h, param2)-2*function(param1, param2)+function(param1-h, param2))/h**2
    matrix[0,1] = matrix[1,0] = (function(param1+h, param2+h)-function(param1+h, param2-h)-
          function(param1-h, param2+h)+function(param1-h, param2-h))/4*h**2 
    matrix[1,1] = (function(param1, param2+h)-2*function(param1, param2)+function(param1, param2-h))/h**2
    
    error_matrix = np.linalg.inv(matrix)
    
    return error_matrix