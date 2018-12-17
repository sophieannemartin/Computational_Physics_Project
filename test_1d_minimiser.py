"""
TESTING 1D MINIMISER ON THE COSHX FUNCTION
05/12/18
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
                x_range, cosh)
    
    
    x3_mins = []
        
    for x3 in x3_list:
        val = cosh(x3)
        x3_mins.append(val)
            
            
    plt.plot(x_range, y_values, label='cosh(x)')
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.title('Minimisation of cosh(x)')
    plt.grid()
    plt.plot(x3_list, x3_mins, '.', color='red', label='minimum')
    plt.legend(prop={'size':14})
    plt.show()
    
if __name__ == "__main__":
    main()