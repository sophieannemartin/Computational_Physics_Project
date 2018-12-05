#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:54:05 2018

@author: sophie
"""

import functions as funcs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

x_range = np.linspace(-10,10, 100)
y_values = funcs.cosh(x_range)

plt.plot(x_range, y_values)
plt.xlabel('x')
plt.ylabel('y')
plt.title('cosh(x)')
plt.grid()
plt.show()