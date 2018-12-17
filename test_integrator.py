"""
TESTING THE INTEGRATOR ON A KNOWN FUNCTION 3X^2 
15/12/18
@author: SOPHIE MARTIN
"""

import numpy as np
import matplotlib.pyplot as plt
import define_functions as functions

x_range = np.linspace(0,1,10)

def func(x):
    return 3*(x**2)

area = functions.integrate_simpson(x_range.min(), x_range.max(), func, n=len(x_range))
y = func(x_range)
plt.plot(x_range, y, label='3x$^2$')
plt.title('Calculated area= %f' % area)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


