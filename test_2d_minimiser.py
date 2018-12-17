"""
TESTING THE 2D MINIMISATION ALGORITHM ON A WELL DEFINED TEST CASE
10/12/18
@author: SOPHIE MARTIN
"""

import numpy as np
import matplotlib.pyplot as plt
import minimiser
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
import time
    
    
def test_f(x,y):
    
    # A test function (can plot in 3D) with a max/min at (0,0)
    f = x**2 + y**2
    return f

def main():
    
    x0, y0 = 2.0,4.0
    
    minimum, minimum_list, iterations = \
    minimiser.minimise_quasi_newton(x0, y0, test_f, 1, 0.01)

    x_range = y_range = np.linspace(-10,10, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    zs = np.array([test_f(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.plot([minimum[0,0]],[minimum[1,0]], [test_f(minimum[0,0], minimum[1,0])],
            markerfacecolor='yellow', markeredgecolor='yellow', marker='o', 
            markersize=3, alpha=1)
    
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Testing the Quasi-Newton 2D minimiser')
    
    
if __name__ == "__main__":
    main()