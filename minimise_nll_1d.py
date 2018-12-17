"""
PLOTTING THE NLL FIT AS A FUNCTION OF TAU
04/12/18
@author: SOPHIE MARTIN
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
    
    # Find the value of the minimum
    minimum, iterations, x3_list = minimiser.minimise_1D(
            taus_range, decayfunction.get_nll_values, maxiter=1000)
    
    nll_mins = []
    
    # Compute the value of the NLL found from each iteration value x3
    for tau in x3_list:
        val = decayfunction.get_nll_values(tau)
        nll_mins.append(val)
    
    # Calculates an error using the value of the NLL with a shift of 0.5
    print('Calculating standard deviation using interpolation...')
    root1, root2 = minimiser.find_standard_deviation(minimum, 
                                                         decayfunction.get_nll_values)
    
    # Calculates the std dev using the curvature of the previous value
    gauss_sigma = minimiser.gauss_standard_deviation(x3_list, 
                                                    decayfunction.get_nll_values)
    
    print('tau-: %f, tau+: %f' % (root1, root2))
    print('sigma-: %f, sigma+: %f' % (minimum-root1, root2-minimum))
    print('Gauss method sigma: ', gauss_sigma)
    
    plt.figure(1)
    plt.plot(taus_range, nll_values)
    plt.plot(x3_list[:-1], nll_mins[:-1], '.', color='red', label='Minimum iterations')
    plt.plot(x3_list[-1], nll_mins[-1], '.', color='green', label='Minimum found')
    plt.xlabel('$\\tau$ (ps)', fontsize=15)
    plt.ylabel('NLL value', fontsize=15)
    plt.grid()
    plt.title('NLL($\\tau$)')
    plt.legend(prop={'size':14})
    

    plt.figure(2)
    plt.plot(taus_range, nll_values)
    plt.xlabel('$\\tau$ (ps)', fontsize=15)
    plt.ylabel('NLL value', fontsize=15)
    plt.grid()
    plt.title('NLL($\\tau$)')
    
    smaller_range = np.linspace(0.2, 0.8, 1000)
    nll_2 = decayfunction.get_nll_values(smaller_range)
    
    plt.figure(3)
    plt.plot(smaller_range, nll_2)
    plt.plot(x3_list[:-2], nll_mins[:-2], '.', color='red', label='Minimum iterations')
    plt.plot(x3_list[-1], nll_mins[-1], '.', color='green', label='Minimum found')
    plt.xlabel('$\\tau$ (ps)', fontsize=15)
    plt.ylabel('NLL value', fontsize=15)
    plt.title('NLL($\\tau$) near minimum')
    plt.grid()
    plt.xlim(0.38,0.43)
    plt.ylim(6215,6240)
    plt.legend(prop={'size':14})
    plt.show()

    return minimum, iterations, x3_list

if __name__ == "__main__":
    minimum, iterations, x3_list = main()