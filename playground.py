"""
INVESTIGATION OF FUNTINS
16/12/18
@author: SOPHIE MARTIN
"""

import define_functions as f
import matplotlib.pyplot as plt
import numpy as np
import minimiser

def function(x):
    # roots x=-5 and x=2
    y = (x**2)+(3*x)-10
    return y

x_range = np.linspace(-10,10, 50)

root1 = minimiser.secant_root_finder([-6,-4], function)
print(root1)
root2 = minimiser.secant_root_finder([1.5,3], function)
print(root2)




minimum, iterations, x3_list = minimiser.minimise_1D(
        taus_range, decayfunction.get_nll_values, maxiter=1000)

plt.plot(x_range, function(x_range))
plt.plot(root1, function(root1), 'o', color='red', label='x=-5')
plt.plot(root2, function(root2), 'o', color='green', label='x=2')
plt.legend(prop={'size':14})
plt.grid()
plt.xlabel('x', fontsize=15)
plt.ylabel('f(x)', fontsize=15)
plt.title('Testing the secant method root finder')

m_range = np.linspace(10,1e6, 1000)
plt.figure()
plt.plot(m_range, 1/np.sqrt(m_range))
plt.show()