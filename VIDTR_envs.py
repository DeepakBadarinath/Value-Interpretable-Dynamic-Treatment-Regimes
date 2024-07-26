#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:51:11 2024

@author: badarinath
"""

'''
Tests VIDTR : Goal environment
'''
import numpy as np
import matplotlib.markers as mrk
import matplotlib.pyplot as plt
import inspect

import disjoint_box_union
from disjoint_box_union import DisjointBoxUnion as DBU
import itertools
from itertools import product, combinations

#%%

class GridEnv:
    
    def __init__(self, dimensions, center, side_lengths, goal, stepsizes=1,
                 initial_state = []):
        '''
        Code for a d dimensional Grid environment where we know the center,
        side_lengths, goal, stepsizes, and initial state. 
        
        In the absence of knowledge of the initial state, we randomly sample from 
        the grid uniformly
        
        Parameters:
        -----------------------------------------------------------------------
        dimensions : int
                     The dimension of the grid
        
        center : np.array
                 The centre of the grid
        
        side_lengths : np.array
                       The side lengths for the grid
        
        goal : np.array
               The goal for the grid
        
        stepsizes : np.array
                    The stepsizes for the grid
        
        initial_state : np.array
                        The initial state in the grid environment
        
        
        Stores:
        -----------------------------------------------------------------------
        state_space : DisjointBoxUnion
                      1 box of dimension self.dimensions, side lengths self.side_lengths,
                      self.center and self.stepsizes
        '''
        
        self.dimensions = dimensions
        self.center = np.array(center)
        
        if type(side_lengths) == int or type(side_lengths) == float:
            self.side_lengths = np.array([[side_lengths for d in range(dimensions)]])
        else:
            self.side_lengths = np.array(side_lengths)
        
        self.goal = np.array(goal)
        
        if type(stepsizes) == int or type(stepsizes) == float:
            self.stepsizes = np.array([stepsizes for d in range(dimensions)])
        else:
            self.stepsizes = np.array(stepsizes)
        
        if len(initial_state) == 0:
            initial_state = GridEnv.sample_point_from_grid(self.center,
                                                           self.side_lengths,
                                                           self.stepsizes)
            print(f'The initial state is {initial_state}')
        
        self.state_space = DBU(1, self.dimensions, self.side_lengths,
                               self.center, self.stepsizes)
        
        actions = []
        for i in range(self.dimensions):
            
            coord_vector = np.zeros(dimensions)
            coord_vector[i] = 1
            
            actions.append(coord_vector)
            actions.append(-coord_vector)
        
        self.actions = actions
        
    def is_within_grid(self, state):
        return all(self.center[d] - self.side_lengths[d]/2 <= state[d] < self.center[d] + self.side_lengths[d]/2 for d in range(self.dimensions))
    
    @staticmethod
    def find_non_zero_index(arr):
        for i, val in enumerate(arr):
            if val != 0:
                return i
        
        return None
    
    @staticmethod
    def sample_point_from_grid(centers, lengths, stepsizes):
        d = len(centers)
        sampled_point = np.empty(d)
        
        for i in range(d):
            start = centers[i] - lengths[i] / 2
            end = centers[i] + lengths[i] / 2
            
            possible_values = list(np.arange(start, end + stepsizes[i], stepsizes[i]))
            sampled_point[i] = np.random.choice(possible_values)
            
        return np.array(sampled_point)
    
    def transition(self, final_state, initial_state, action,
                   state_space=[], action_space=[]):
        
        initial_state = np.array(initial_state)
        final_state = np.array(final_state)
        
        if not self.is_within_grid(initial_state) or not self.is_within_grid(final_state):
            return 0
        
        expected_final_state = initial_state + action * self.stepsizes[GridEnv.find_non_zero_index(action)]
        if np.array_equal(expected_final_state, final_state):
            return 1
    
        return 0
    
    def move(self, initial_state, action):
        
        possible_new_state = np.array(initial_state) + action * np.array(self.stepsizes[GridEnv.find_non_zero_index(action)])
        
        if self.is_within_grid(possible_new_state):
            return possible_new_state
        
        else:
            return initial_state
    
    def plot_policy_2D(self, policy,
                       title = 'Policy directions',
                       arrow_length = 0.2):
        
        x = np.arange(self.center[0] - self.side_lengths[0]/2,
                      self.center[0] + self.side_lengths[0]/2 + 1,
                      self.stepsizes[0])
        
        y = np.arange(self.center[1] - self.side_lengths[1]/2,
                      self.center[1] + self.side_lengths[1]/2 + 1,
                      self.stepsizes[1])
        
        goal = self.goal
        
        xx,yy = np.meshgrid(x,y)
        
        print('X values are')
        print(xx)
        
        print('Y values are')
        print(yy)
        
        output_grid = np.empty((len(x), len(y)), dtype = object)
        
        for i,el_x in enumerate(x):
            for j,el_y in enumerate(y):
                
                output_grid[i, j] = policy(np.array([el_x,el_y]))
                    
        print('The outputs look like')
        print(output_grid)
        
        # Plotting the outputs
        fig, ax = plt.subplots()
        ax.set_xlim(self.center[0] - self.side_lengths[0]/2, self.center[0] + self.side_lengths[0]/2)
        ax.set_ylim(self.center[1] - self.side_lengths[1]/2, self.center[1] + self.side_lengths[1]/2)
        ax.set_xticks(x)
        ax.set_yticks(y)
        ax.grid(True)
        
        # Assign directions to the function outputs
        direction_dict = {
            (1, 0): 'right',
            (-1, 0): 'left',
            (0, 1): 'up',
            (0, -1): 'down'
        }
        
        # Plot the directions
        for i,el_x in enumerate(x):
            for j,el_y in enumerate(y):
                direction = direction_dict[tuple(output_grid[i, j])]
                if direction == 'right':
                    ax.arrow(el_x, el_y, arrow_length, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')
                elif direction == 'left':
                    ax.arrow(el_x, el_y, - arrow_length, 0, head_width=0.2, head_length=0.2, fc='g', ec='g')
                elif direction == 'up':
                    ax.arrow(el_x, el_y, 0, arrow_length, head_width=0.2, head_length=0.2, fc='b', ec='b')
                elif direction == 'down':
                    ax.arrow(el_x, el_y, 0, -arrow_length, head_width=0.2, head_length=0.2, fc='y', ec='y')
        
        plt.scatter(goal[0], goal[1], marker = 'x', label = 'Goal')
        
        plt.gca().invert_yaxis()
        ax.set_title(title)
        plt.legend()
        plt.show()
        
    def reward(self, state, action,
               states = [],
               actions = []):                                            
        
        inv_distance = 1/ (np.linalg.norm(self.move(state, action) - self.goal) + 1)               
        
        return inv_distance
    
    def plot_policy_reward(self,
                           policy,
                           title='Reward function when evaluated over 2D grid'):
        
        x = np.arange(self.center[0] - self.side_lengths[0]/2,
                      self.center[0] + self.side_lengths[0]/2,
                      self.stepsizes[0])
        
        y = np.arange(self.center[1] - self.side_lengths[1]/2,
                      self.center[1] + self.side_lengths[1]/2,
                      self.stepsizes[1])
        
        X,Y = np.meshgrid(x,y)
        
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = np.array([X[i,j], Y[i,j]])
                action = policy(state)
                Z[i,j] = self.reward(state, action)
        
        plt.figure(figsize=(8,6))
        contour = plt.contour(X, Y, Z, cmap = 'viridis')
        plt.colorbar(contour)
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()