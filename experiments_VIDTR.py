#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:00:29 2024

@author: badarinath
"""
# Experimental setup for Bosul-basal+bolus-none

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#%%

class Scenario1:
    '''
    We simulate the scenario 1 corresponding the IDTR paper.
    
    Here the states, actions, and rewards are independent of time.
    '''
    
    def __init__(self):
        
        self.states = [np.random.normal(size = 50), np.array([])]
        self.actions = [np.random.choice([-1,1]), np.random.choice([-1,1])]
        
        rewards_1 = np.random.normal(loc = 0.5 * self.states[0][2] * self.actions[0])
        rewards_2 = np.random.normal(loc=(self.states[0][0]**2 + self.states[0][1]**2 - 0.2)*
                                         (0.5 - self.states[0][0]**2 - self.states[0][1]**2) + rewards_1)
        
        self.rewards = [rewards_1, rewards_2] 
        
        

class Scenario2:
    '''
    We simulate the scenario 2 corr to the IDTR paper.
    
    '''
    
    def __init__(self):
        
        scenario1 = Scenario1()
        
        self.states = []
        self.actions = scenario1.actions
        
        self.states.append(scenario1.states[0])
        states2 = np.array([np.random.binomial(1, 1 - norm.cdf(1.25 * self.states[0][0] * self.actions[0])),
                            np.random.binomial(1, 1 - norm.cdf(-1.75 * self.states[0][1] * self.actions[0]))])
        
        self.states.append(states2)
        
        rewards_1 = np.random.normal(loc = 1.5 * self.states[0][2] * self.actions[0])
        rewards_2 = np.random.normal(loc = (0.5 + rewards_1 + 0.5 * self.actions[0] + 0.5 * states2[0] - 0.5 * states2[1])*self.actions[1])
        
        self.rewards = [rewards_1, rewards_2]
    
    
class Scenario3:
    '''
    We simulate the third scenario corr to the IDTR paper.
    '''
    
    def __init__(self):
        
        self.actions = [np.random.choice([-1,1]), np.random.choice([-1,1]), np.random.choice([-1,1])]
        
        states1 = np.array([np.random.normal(45, scale=15.0),
                            np.random.normal(45, scale=15.0),
                            np.random.normal(45, scale=15.0)])
        
        states2 = np.random.normal(1.5 * states1[0], scale = 10.0)
        states3 = np.random.normal(0.5 * states2, scale = 10.0)
        
        rewards1 = 0
        rewards2 = 0
        rewards3 = np.random.normal(20 - 10.6 * states1[0] -
                                         40 * (float(self.actions[0]>0) - (states1[0] > 30))**2
                                         - 10.8 * states2 - 60 * (float(self.actions[1]>0) - float(states2>40))**2
                                         - np.abs(1.4 * states3 - 40) * (float(self.actions[2]>0) - float(states3 > 40))**2)
        
        self.states = [states1, states2, states3]
        self.rewards = [rewards1, rewards2, rewards3]


class Scenario4:
    '''
    We simulate the fourth scenario corr to the IDTR paper.
    '''
    def __init__(self):
        
        self.states = np.array([np.random.normal(45, scale = 15.0, size=50),
                                np.random.normal(45, scale=15.0, size=50),
                                np.random.normal(45, scale = 15.0, size = 50)])
        
        scenario3 = Scenario3()
        self.actions = scenario3.actions
        self.rewards = scenario3.rewards


class Scenario5:
    '''
    We simulate the fifth scenario corr to the IDTR paper
    '''
    def __init__(self):
        
        actions = []
        for t in range(10):
            a1 = np.random.choice([0,1])
            if a1 == 0:
                a2 = np.random.choice([0,1,2,3])
            else:
                a2 = np.random.choice([1,2,3])
            actions.append((a1, a2))
        
        self.actions = actions
        u_vals = np.random.normal(0, scale=0.1, size = 10)
        
        self.states = []
        self.rewards = []
        self.states.append(0.5 + u_vals[0])
        for t in range(1,10):
            
            self.states.append(0.5 + 0.2 * self.states[t-1]
                               - 0.07 * self.actions[t-1][0] * self.actions[t-1][1]
                               -0.01 * (1-self.actions[t-1][0]) * self.actions[t-1][1] + u_vals[t])
            
            self.rewards.append(np.random.normal(30 * float(t==0) - 5 * u_vals[t] - 6 * (actions[t][0] - (self.states[t] > 5/9))**2 - 1.5 * actions[t][0] * (actions[t][0] - 2 * self.states[t]) - 1.5 * (1 - self.actions[t][0])*(self.actions[t][1] - 5.5 * self.states[t])**2,
                                                 scale = 0.8))

#%%


# [state[i], action[i], state[i+1]] for i in range(T-1)

def get_trajectory(scenario):
    
    for i in range():
        
        
    return 


#%%

s1 = Scenario1()
print('States corresponding to scenario 1 is')
print(s1.states)

print('Actions corresponding to scenario 1 is')
print(s1.actions)

print('Rewards corr to scenario 1 is')
print(s1.rewards)


#%%

s2 = Scenario2()
print('States corresponding to scenario 2 is')
print(s2.states)

print('Actions corresponding to scenario 2 is')
print(s2.actions)

print('Rewards corr to scenario 2 is')
print(s2.rewards)

#%%

s3 = Scenario3()
print('States corresponding to scenario 3 is')
print(s3.states)

print('Actions corresponding to scenario 3 is')
print(s3.actions)

print('Rewards corr to scenario 3 is')
print(s3.rewards)


#%%

s4 = Scenario4()
print('States corresponding to scenario 4 is')
print(s4.states)

print('Actions corresponding to scenario 4 is')
print(s4.actions)

print('Rewards corr to scenario 4 is')
print(s4.rewards)

#%%

s5 = Scenario5()
print('States corresponding to scenario 5 is')
print(s5.states)

print('Actions corresponding to scenario 5 is')
print(s5.actions)

print('Rewards corresponding to scenario 5 is')
print(s5.rewards)