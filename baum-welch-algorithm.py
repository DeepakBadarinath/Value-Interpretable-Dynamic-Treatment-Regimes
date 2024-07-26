import numpy as np
import matplotlib.pyplot as plt

#%%
class BaumWelch:
    
    def __init__(self, dimension, trajectories, actions,
                 bounds = [], differences = []):
        
        '''
        We design the class for the Baum Welch algorithm where we are given the
        trajectories and we wish to design the estimate for the transitions and
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
        
        actions : list
                  The list of actions we can take in the MDP
        
        bounds : np.array[2, d]
                 The bounds in each dimension of the state space
    
        differences : np.array[d]
                      The differences in each dimension of the state space for
                      the Baum Welch algorithm
        
        Stores:
        -----------------------------------------------------------------------
        state_sizes : np.array[d]
                     We calculate the state size after having knowledge of the 
                     bounds, differences, etc
        
        no_of_trajs : int
                      The number of trajectories that is given
        '''
        self.trajectories = trajectories
        self.no_of_trajs = len(trajectories)
        self.dimension = dimension
        self.actions = actions
        self.bounds = bounds
        
        if len(differences) == 0:
            self.differences = np.ones(self.dimension)
        
        if len(bounds) == 0:
            self.bounds = self.calculate_upper_lower_bounds()
        
        state_sizes = []
        for i in range(self.dimension):
            state_sizes.append(self.bounds[1, i] - self.bounds[0, i])
        
        self.state_sizes = np.array(state_sizes)
        

    def calculate_upper_lower_bounds(self):
        '''
        Calculate the upper and lower bounds for each dimension of the state space
        
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
            category.append((s[d] - self.bounds[0,d]) / self.differences[d] + 1)
        

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
            category_count.append((self.bounds[1,d] - self.bounds[0,d])/self.differences[d] + 1)
        
        self.category_count = np.array(category_count)
        
        for s, a, s_next, r in self.trajectories:
            for d in range(self.dimension):
                               
                cat_traj.append( [self.category_function(s), a, self.category_function(s_next), r] )
        
        return cat_traj 

