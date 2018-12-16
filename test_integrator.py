#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 14:04:57 2018

@author: sophie

Testing the integrator on a known function 3*x*2
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


