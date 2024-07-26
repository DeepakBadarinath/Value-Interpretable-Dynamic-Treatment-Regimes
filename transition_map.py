#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:06:45 2024

@author: badarinath
"""

def transition_map(W, H, cx, cy):
    def is_within_grid(x, y):
        return 0 <= x < W and 0 <= y < H

    def transition(initial_state, final_state, action):
        ix, iy = initial_state
        fx, fy = final_state
        
        if not is_within_grid(ix, iy) or not is_within_grid(fx, fy):
            return 0

        if action == 0:  # Up
            if fx == ix and fy == iy + 1:
                return 1
        elif action == 1:  # Right
            if fx == ix + 1 and fy == iy:
                return 1
        elif action == 2:  # Down
            if fx == ix and fy == iy - 1:
                return 1
        elif action == 3:  # Left
            if fx == ix - 1 and fy == iy:
                return 1
        
        return 0

    return transition

# Example usage
W = 5  # Grid width
H = 5  # Grid height
cx, cy = 2, 2  # Center of the grid

# Create the transition function for the given grid
transition = transition_map(W, H, cx, cy)

# Test the transition function
initial_state = (2, 2)
final_state = (2, 3)
action = 0  # Up
print(transition(initial_state, final_state, action))  # Output: 1

final_state = (3, 2)
action = 1  # Right
print(transition(initial_state, final_state, action))  # Output: 1

final_state = (2, 1)
action = 2  # Down
print(transition(initial_state, final_state, action))  # Output: 1

final_state = (1, 2)
action = 3  # Left
print(transition(initial_state, final_state, action))  # Output: 1

final_state = (2, 4)
action = 0  # Up
print(transition(initial_state, final_state, action))  # Output: 0
