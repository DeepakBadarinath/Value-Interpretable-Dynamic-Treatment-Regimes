#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:37:48 2024

@author: badarinath
"""
import numpy as np
import matplotlib.pyplot as plt
import baum_welch_algorithm as bw
from importlib import reload
import inspect


bw = reload(bw)
from baum_welch_algorithm import BaumWelch

#%%
'''
Tests for Baum Welch
'''
dimension = 2
trajectories = [[np.array([1,0]), -1, np.array([0,2]), -10],
                [np.array([0,0]), 1, np.array([0,1]), -4],
                [np.array([1,0]), 0, np.array([0,-1]), 10],
                [np.array([-1,0]), -1, np.array([0,0]), 5],
                [np.array([1,0]), 1, np.array([0,1]), 3],
                [np.array([0,0]), -1, np.array([0,0]), 10]]

actions = [-1, 0, 1]

algo = BaumWelch(dimension, trajectories, actions, 4)


print('Upper and Lower bounds are')
print(algo.calculate_upper_lower_bounds())

print('Categories are')
print(algo.calculate_categories())
print('Category function evaluated on [0,1], [1,0], and [3,6] is')
print(algo.category_function(np.array([0,1])))
print(algo.category_function(np.array([1,0])))
print(algo.category_function(np.array([3,6])))

print('Categories where we do the evaluation are')                                  
print(algo.calculate_categories())                                                          


f = lambda s, action, state_space, action_space: 0.0
print('Zero transitions evaluated at [0,1] for the function that changes at that point')
print(BaumWelch.increment_total_occurences_by_one(f, np.array([0,1]), 1, None, actions)(np.array([0,1]), 1))
                                                                                
print('Zero transitions evaluated at [5,0] for a function that does not change at that point')
print(BaumWelch.increment_total_occurences_by_one(f, np.array([0,0]), 0, None, actions)(np.array([0,0]), 0))
                                                                                        
                                                                                            
g = lambda s_new, s, a, state_space, action_space : 0.0                         

print('Increment transitions evaluated at ')
print(BaumWelch.increment_transitions_by_one(g, np.array([0,0]), np.array([0,1]), 1.0, None, actions)(np.array([0,0]), np.array([0,1]), 1.0))
print(BaumWelch.increment_transitions_by_one(g, np.array([0,1]), np.array([0,0]), -1.0, None, actions)(np.array([0,1]), np.array([0,0]), -1.0))


print('Baum Welch transitions are')
print(algo.estimate_transitions()[0](np.array([0,0]), np.array([0,1]), -1))     
print(algo.estimate_transitions()[1](np.array([0,1]), np.array([0,0]), 1))      
