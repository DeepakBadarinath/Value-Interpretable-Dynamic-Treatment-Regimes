#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:11:52 2024

@author: badarinath
"""

import numpy as np
import itertools
from itertools import product

coord_list = []
coord_list.append(np.arange(0, 1, 0.1))
coord_list.append(np.arange(0, 1, 0.2))

integral = 0.0

f = lambda x: np.sum(x**2)

for p in itertools.product(*coord_list):
    integral =  f(np.array(p)) + integral

print(integral)