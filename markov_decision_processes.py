#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 09:53:59 2024

@author: badarinath
"""
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats


class MarkovDecisionProcess:
    
    '''
    Class that represents a Markov Decision Process, with the states, actions,
    transitions and rewards
    '''
    
    def __init__(self, states, actions, time_horizon, gamma, transition_kernels,
                 reward_functions):
        
        '''
        Parameters:
        -----------------------------------------------------------------------
        states : Representation of the state space of the MDP (usually set or DisjointBoxUnion)
                 The state space of the MDP                                    
         
        actions : Representation of the action space of the MDP (usually set)  
                  The actions for the MDP                                      
                  
        time_horizon : int                                                     
                       The time horizon for the MDP                            
        
        gamma : float                                                          
                The discount factor for the MDP
        
        transition_kernels : list[function(state, state, action, state_space, action_space) \to [0,1]]
                             List of length T which consists of probability
                             transition maps.
                             Here the sum of the transition_kernels(s',s,a) for
                             all s' in states = 1
        
        reward_functions : list[function(state, action, state_space, action_space) \to \mathbb{R}]
                           List of length T which consists of reward_functions 
        
        '''
        
        self.states = states
        self.actions = actions
        self.time_horizon = time_horizon
        self.gamma = gamma                                                     
        self.transition_kernels = transition_kernels                           
        self.reward_functions = reward_functions                                
                                                                                        
                                                                                    
                                                                                
                                                                                        
                                                                                        