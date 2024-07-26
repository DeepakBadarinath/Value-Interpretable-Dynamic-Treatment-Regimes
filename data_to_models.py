#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:43:49 2024

@author: badarinath
"""

'''
In this .py file we lay out the implementation of going from the existing data
to the probabilistic models that determine the parameters in the MDP
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy

'''
Given trajectories from an MDP, find the probability transition kernel and the 
reward functions for the different timesteps, states, and the rewards
'''
