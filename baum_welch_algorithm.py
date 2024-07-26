import numpy as np
import matplotlib.pyplot as plt

# To do - tasks till Friday 16:30
# Test each feature after it has been added separately

###############################################################################

# To do:
    # 1. Code up Discretized BaumWelch
    # 2. Integrate BaumWelch with IDTR data to get transitions and rewards
    # 3. Use the obtained transitions and rewards for the VIDTR algorithm 

# Tuesday                                                               
# BaumWelch add features - probability transitions and reward estimations
# BaumWelch test on homogeneous state space synthetic data              
# Tests                                                                   

# Wednesday                                                              
# BaumWelch reward discretization                                        
# BaumWelch on VIDTR data to get transitions and rewards                 
# VIDTR on new data - compute results                                     

# Thursday
# Data engineering:
# Change existing trajectory data to encorporate discretization
# BaumWelch on real world data
# Engineer

#%%
class BaumWelch:
    
    def __init__(self, dimension, trajectories, actions, time_horizon,
                 bounds = [], state_differences = [], reward_differences = [],
                 state_space = None, action_space = None):
        
        '''
        We design the class for the Baum Welch algorithm where we are given the
        trajectories and we wish to design the estimates for the transitions and
        the rewards.

        Parameters:
        -----------------------------------------------------------------------
        dimension : int
                    The dimension of the MDP state space
        
        trajectories : list[no_of_trajs]
                       A list of size N which represents the different trajectories 
                       that we use to estimate the transitions and the rewards
                        
                       Each trajectory here looks like 
                       [(state[i], action[i], state[i+1], reward[i])]
                       here i = 0, 1, 2,....,T-1
                       
                       Example T1 : [[np.array([0,0]), 1.0, np.array([0,1]), 10.0]]
                                       s_0             a_0        s_1         r_1
                       
        actions : list
                  The list of actions we can take in the MDP
        
        bounds : np.array[2, d]
                 The bounds in each dimension of the state space
    
        state_differences : np.array[d]
                            The differences in each dimension of the state space for
                            the Baum Welch algorithm
        
        reward_differences : np.array[d]
                             The differences in each timestep for the rewards
        
        time_horizon : int
                       The time horizon of the trajectory
        
        state_space : domain of the transitions
                      Domain of the MDP
                       
        Stores:
        -----------------------------------------------------------------------
        state_sizes : np.array[d]
                     We calculate the state size after having knowledge of the 
                     bounds, differences, etc
        
        no_of_trajs : int
                      The number of trajectories that is given
                      
        transitions : list[function(s_new, s, action, state_space, action_space) \to [0,1]]
                      List of length T which consists of the number of transitions we observe 
                      while going from s -> s' after taking action with the given state_space and action_space
        
        '''
        self.trajectories = trajectories
        self.no_of_trajs = len(trajectories)
        self.dimension = dimension
        self.actions = actions
        self.bounds = bounds
        self.time_horizon = time_horizon
        
        if len(state_differences) == 0:
            self.state_differences = np.ones(self.dimension)
        
        if len(reward_differences) == 0:
            self.reward_differences = np.ones(self.dimension)
        
        if len(bounds) == 0:
            self.bounds = self.calculate_upper_lower_bounds()
        
        print(self.bounds)
        
        state_sizes = []
        for i in range(self.dimension):
            state_sizes.append(self.bounds[1, i] - self.bounds[0, i])
        
        self.state_sizes = np.array(state_sizes)
        
        zero_transition = lambda s_new, s, action, state_space, action_space : 0
        self.transitions = [zero_transition for i in range(self.time_horizon)]
        
        zero_occurence = lambda s, action, state_space, action_space : 0
        self.total_occurences = [zero_occurence for i in range(self.time_horizon)]
        
        self.transition_probs = [zero_transition for i in range(self.time_horizon)]
        
        self.state_space = state_space
        
    def calculate_upper_lower_bounds(self):
        '''
        Calculate the upper and lower bounds for each dimension of the state space and
        add the bounds for the rewards
        
        Stores:
        -----------------------------------------------------------------------
        bounds : np.array[2,d]
                 Upper and lower bounds 
        '''
        state_bounds = np.array([[+np.inf, -np.inf] for d in range(self.dimension)])
        
        for s, a, s_next, r in self.trajectories:
            for d in range(self.dimension):
               
                if s[d] > state_bounds[d,1]:
                    state_bounds[d,1] = s[d]
                
                
                if s[d] < state_bounds[d,0]:
                    state_bounds[d,0] = s[d]
                    
        
        for d in range(self.dimension):
            if self.trajectories[-1][2][d] > state_bounds[d,1]:
                state_bounds[1,d] = self.trajectories[-1][2][d]
            
            if self.trajectories[-1][2][d] < state_bounds[d,0]:
                state_bounds[0,d] = self.trajectories[-1][2][d]
         
        return state_bounds            
    
    def category_function(self, s):
        '''
        Given the upper/lower bounds and the dimension of the state space,
        calculate the category.
        
        Returns:
        -----------------------------------------------------------------------
        category : np.array[d]
                   The category of the point

        '''
        category = []
        for d in range(self.dimension):
            if s[d] >= self.bounds[0,d] and s[d] <= self.bounds[1,d]:
                category.append((s[d] - self.bounds[0,d]) / self.state_differences[d] + 1)
            
            elif s[d] < self.bounds[0,d]:
                category.append(0)
            
            else:
                category.append((self.bounds[1,d] - self.bounds[0,d]) / self.state_differences[d] + 1)

        return np.array(category)
    
    
    def calculate_categories(self):
        '''
        Given the trajectory data, calculate the categories for the given state
        bounds. 
        
        Stores:
        -----------------------------------------------------------------------
        category_count : np.array
                         The number of elements we have for each dimension
        
        cat_traj : list[N]
                   The different categories for the BaumWelch algorithm
                
        '''
        cat_traj = []
        category_count = []
        
        for d in range(self.dimension):
            category_count.append((self.bounds[1,d] - self.bounds[0,d])/self.state_differences[d] + 2)
        
        self.category_count = np.array(category_count)
        
        for s, a, s_next, r in self.trajectories:
            for d in range(self.dimension):
                               
                cat_traj.append( [self.category_function(s), a,
                                  self.category_function(s_next), r] )
        
        return cat_traj 
    
    @staticmethod
    def increment_transitions_by_one(f, s_new, s, a,
                                     state_space=None, action_space=None):
        '''
        Given a function f(s_2, s_1, a_1, state_space, action_space), redefine it
        such that f(s_new, s, a, state_space, action_space) is incremented by 1
                                                                               
        Parameters:
        -----------------------------------------------------------------------
        f : function(s2, s1, a)                                                
            Old function we wish to redefine                                       
             
        s_new : type(domain(function)[0])                                      
                The new state at which we wish to redefine f                   
         
        s : type(domain(function)[1])                                          
            The old state at which we wish to redefine f                       
            
        a : type(range(function)[2])                                           
            The action at which we wish to redefine f                            
        
        state_space : type(domain(function))                                   
                      The state space of the function                          
        
        action_space : type(range(function))                                   
                       The range space of the function                         
        
        Returns:
        -----------------------------------------------------------------------
        g : function
            New function which is such that g(s_new, s, a) = f(s_new, s, a) + 1

        '''
        def g(s_2, s_1, a_1, state_space=None, action_space=None):
            if np.sum((s_2 - s_new)**2 + (s_1 - s)**2 + (a-a_1)**2) == 0:
                return f(s_new, s, a, state_space, action_space) + 1            
            else:
                return f(s_2, s_1, a_1, state_space, action_space)
        return g
    
    @staticmethod
    def increment_total_occurences_by_one(f, s, a,
                                          state_space=None, action_space=None):
        '''
        Given a function f(s,a,state_space,action_space), redefine it such that

        Parameters:
        -----------------------------------------------------------------------
        f : function(s1, a)
            Old function we wish to redefine
            
        s : type(domain(function)[0])
            The new state at which we wish to redefine f
            
        a : type(domain(function)[1])
            The new action at which we wish to redefine f                      
            
        state_space : type(domain(function))
                      The state space of the function f
            
        action_space : type(range(function))
                       The action space of the function f 

        Returns:
        -----------------------------------------------------------------------
        g : function(s1, a, state_space, action_space)                         
            The new function g is such that it is same as f on all points but (s,a)
            g(s,a)

        '''
        def g(s1, a1, state_space=None, action_space=None):
            if np.sum((s1-s)**2 + (a-a1)**2 ) == 0:
                return f(s, a, state_space, action_space)
            else:
                return f(s1, a1, state_space, action_space)
        
        return g
                                                                                
    def estimate_transitions(self):
        '''
        Given the trajectory data, estimate the transition probability function.
        
        Stores:
        -----------------------------------------------------------------------
        transitions : list[function(s_new, s, action, state_space, action_space)]
                      The number of transitions we perform in which we move from 
                                                                                  
        '''
        for t in range(self.time_horizon):                                      
            for traj_no, traj in enumerate(self.trajectories):                             
                
                s_new = traj[2*t+2]
                s = traj[2*t]
                a = traj[2*t+1]
                                                                                
                self.transitions[t] = BaumWelch.increment_transitions_by_one(self.transitions[t],
                                                                             s_new, s, a,
                                                                             self.state_space,
                                                                             self.actions)
        
        for t in range(self.time_horizon):                                      
            for traj_no, traj in enumerate(self.trajectories):                             
                
                s_new = traj[2*t+2]
                s = traj[2*t]
                a = traj[2*t+1]
                
                self.total_occurences[t] = BaumWelch.increment_total_occurences_by_one(self.total_occurences[t],
                                                                                       s, a,
                                                                                       self.state_space,
                                                                                       self.actions)
        
        for t in range(self.time_horizon):
            
            s_new = traj[2*t+2]
            s = traj[2*t]
            a = traj[2*t+1]
            
            def new_transition_probs(t, s_new, s, action, state_space = None, action_space = None):
                
                if self.total_occurences[t](s, action, self.state_space, self.actions) > 0:
                    
                    return self.transitions[t](s_new, s, action, self.state_space, self.actions) / self.total_occurences[t](s, action, self.state_space, self.actions)
                    
                else:
                    
                    return 0
            
            (self.transition_probs)[t] = lambda s_new, s, a : new_transition_probs(t, s_new, s, a)
            
        return self.transition_probs