#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:25:10 2024

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mrk
import VIDTR_envs as envs
from VIDTR_envs import GridEnv
from importlib import reload
import disjoint_box_union
from disjoint_box_union import DBUIterator


envs = reload(envs)
#%%

x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.5)
xx,yy = np.meshgrid(x,y)

markers = {(1,0): mrk.CARETRIGHT,
           (0,1): mrk.CARETUP,
          (-1,0): mrk.CARETLEFT,
          (0,-1): mrk.CARETDOWN}

action_color_dict = {(1,0):'red', (0,1):'blue',
                    (-1,0):'green', (0,-1):'grey'}

side_lengths = np.array([6,6])

goal = np.array([1,0])


def policy(curr_point, goal=goal):
    
    if curr_point[0] < goal[0]:
        direction = np.array([1,0])
    elif curr_point[0] > goal[0]:
        direction = np.array([-1,0])
    
    elif curr_point[1] > goal[1]:
        direction = np.array([0,-1])
    
    else:
        direction = np.array([0,1])

    return direction


environment = GridEnv(dimensions=2,
                      center = np.array([0,0]),
                      side_lengths = side_lengths,
                      goal = goal)

#%%
'''
Tests for transitions and move
'''

print(environment.transition(initial_state = np.array([0,0]),
                             final_state = np.array([1,0]),
                             action = np.array([1,0])))

print(environment.transition(initial_state = np.array([1,0]),
                             final_state = np.array([1,1]),
                             action = np.array([0,1])))

print(environment.move(np.array([0,0]), action = np.array([0,1])))

#%%
'''
Tests for plotting rewards
'''

stepsizes = 0.5
policy = policy

environment.plot_policy_2D(policy)

state_space = environment.state_space
iter_state_class = disjoint_box_union.DBUIterator(environment.state_space)
iter_state = iter(iter_state_class)

for i,s in enumerate(iter_state):
    print(f'{i}th element is {s}')
    a = policy(s)
    print(f'Reward at state {s}, when we take action {a} is {environment.reward(s,a)}')

#%%
'''
Tests for plot_policy_2D
'''
environment.plot_policy_reward(policy, title='Rewards for the final policy')
