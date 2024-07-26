#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# To dos are also listed in this .py file
"""                    
Created on Fri May 24 22:29:40 2024
                       
@author: badarinath
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import constraint_conditions as cc                                             
import interval_and_rectangle_operations as op

from importlib import reload                                                    
from itertools import combinations, product                                    
                                                                                
op = reload(op)                                                                
#%%                                                                            
                                                                                                                                                                 
# Task breakdown:-                                                              
###############################################################################

# Baum Welch and other algorithms to get the models for the data - design and debug              
# VIDTR based on the examples from the IDTR paper                                    
                                                                                
# VIDTR based on the real world data                                             
###############################################################################         
#%%                                                                            
class DisjointBoxUnion:                                                         
                                                                                                                                                    
    '''                                                                        
    Represent a disjoint union of d-dimensional boxes in \mathbb{R}^d.         
    '''

    def __init__(self, no_of_boxes, dimension, lengths, centres, stepsizes = [],
                 store_endpoints = True, store_bounds = True):
        '''
        Parameters:    
        -----------------------------------------------------------------------
            no_of_boxes : int                                                  
                          Number of disjoint boxes                                 

            dimension : int                                                    
                        Dimension of the space we reside in                    

            lengths : np.array(no_of_boxes, dimension)                          
                      A numpy array that denotes the lengths of the different  
                      dimensions                                               
                                                                               
            centres : np.array(no_of_boxes, dimension)                          
                      A numpy array denoting the coordinates of the different  
                      centres of the many boxes                                  
                                                                               
            stepsizes : np.array(no_of_boxes, dimension) or int or float       
                        stepsizes[i, d] = the stepsize for the nth box and the 
                        dth dimension
    
            
        Stores:
        -----------------------------------------------------------------------
            bounds : list(2 * d) of length no_of_boxes
                     The lower and upper bound for each dimension and each box
            
        end_points : np.array(N * 2D * D)
                     The end points of the n'th box each of which has d coordinates
                        
        '''

        self.no_of_boxes = no_of_boxes
        self.dimension = dimension
        
        if len(centres.shape) == 1:
            self.centres = np.expand_dims(centres, axis=0)
        else:
            self.centres = centres
        
        if len(lengths.shape) == 1:
            self.lengths = np.expand_dims(lengths, axis=0)
        else:
            self.lengths = lengths
        
        self.point_count = None
        
        if type(stepsizes) == int or type(stepsizes) == float:
            self.stepsizes = stepsizes * np.ones((no_of_boxes, dimension))
        
        elif np.sum(stepsizes) == 0:
            self.stepsizes = np.ones((no_of_boxes, dimension))
        
        elif len(stepsizes.shape) == 1:
            self.stepsizes = np.expand_dims(stepsizes, axis=0)
        
        else:
            self.stepsizes = stepsizes
        
        if store_bounds:
            self.bounds = self.get_bounds()
        if store_endpoints:
            self.endpoints = self.get_end_points()
    
    
    def no_of_points(self):
        '''
        Given a DBU return the total number of points in it
        '''
        dbu_iter_class = DBUIterator(self)
        dbu_iterator = iter(dbu_iter_class)
        
        no_of_points = 0
        for s in dbu_iterator:
            no_of_points += 1
        
        self.point_count = no_of_points
        return no_of_points
    
    
    def add_disjoint_rectangle(self, dimension, centre, lengths, stepsize=[]):
        '''
        Given a DBU add a new rectangle of the same dimension assuming it does
        not intersect with the existing DBU.
        
        Parameters:
        -----------------------------------------------------------------------
        dimension : int
                    The dimension of the new rectangle
        
        centre : np.array[dimension]
                 The centre of the new rectangle
        
        lengths : np.array[dimension]
                  The lengths of the sides of the new rectangle
        
        stepsize : np.array[dimension]
                   The stepsizes over the various dimensions of the new rectangle.
        
        '''
        
        if dimension != self.dimension:
            print(f'Dimension of new box which is {dimension} != {self.dimension} which is the dimension of the DBU.')
            return
        
        if self.no_of_boxes == 0:
            self.lengths = np.expand_dims(lengths, axis=0)
            self.stepsize = np.expand_dims(np.ones(dimension), axis=0)
            self.centres = np.expand_dims(centre, axis=0)
        
        else:
            self.lengths = np.vstack([self.lengths, lengths])
            stepsize = np.ones(dimension)
            self.stepsizes = np.vstack([self.stepsizes, stepsize])
            self.centres = np.vstack([self.centres, centre])

        self.no_of_boxes += 1        
    
    
    def integrate(self, function):
        '''
        Given a function taking inputs in d dimensions, evaluate the Riemann-Integral 
        of the function over the DBU.

        Parameters:
        -----------------------------------------------------------------------
            function : func
                       The d-dimensional function to integrate over the DBU
                     
        Returns:         
        -----------------------------------------------------------------------
            integral : float
                  The value of the integral of the function over the DBU.
                  
        '''
        
        if self.bounds == None:
            self.get_bounds()
    
        integral = 0.0
        
        coord_list = []
        
        for box_no in range(self.no_of_boxes):
            for dim in range(self.dimension):
                
                coord_list.append(np.arange(self.bounds[box_no][dim][0],
                                            self.bounds[box_no][dim][1],
                                            self.stepsizes[box_no, dim]))         
            
        for point in itertools.product(*coord_list):
            integral = function(np.array(point)) + integral
            
        return integral
            
        
    def subtract_constraint(self, constraint_condition):
        '''
        Given a DBU subtract a constraint condition from it to give another DBU.

        Parameters:
        -----------------------------------------------------------------------
            constraint_condition : type(constraint_conditions)
                                   The constraint condition we wish to subtract
                                   from the DBU

        Returns:
        -----------------------------------------------------------------------
            final_DBU : The final disjoint box union after removing the region
            under the constraint condition
        '''
        final_DBU = DisjointBoxUnion(0, self.dimension,
                                     np.array([[]]), np.array([[]]))
        
        for box_no in range(self.no_of_boxes):
            
            constraint_DBU = constraint_condition.create_a_DBU(self.stepsizes[box_no])
            
            single_box_DBU = DisjointBoxUnion(1, self.dimension,
                                              np.array([self.lengths[box_no]]),
                                              np.array([self.centres[box_no]]),
                                              np.array([self.stepsizes[box_no]]))
            
            subtracted_DBU = single_box_DBU.subtract_single_DBUs(constraint_DBU)
            if np.sum(subtracted_DBU.lengths) > 0:
                final_DBU = final_DBU.append_disjoint_DBUs(subtracted_DBU)

        return final_DBU
    
    
    def subtract_DBUs(self, DBU):
        '''
        Given a DBU self and another DBU, subtract one from the other.
        
        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The DBU we wish to subtract from the original DBU
        
        Returns:
        -----------------------------------------------------------------------
        subtracted_DBU : DisjointBoxUnion
                         The DBU we get after subtraction
        '''
        final_DBU = DisjointBoxUnion.empty_DBU(self.dimension)
        
        for i in range(self.no_of_boxes):
            
            DBU_single = DisjointBoxUnion(1, self.dimension,
                                          np.array([self.lengths[i]]),
                                          np.array([self.centres[i]]))
            
            subtracted_DBU = DBU_single

            for j in range(DBU.no_of_boxes):
                DBU_2 = DisjointBoxUnion(1, self.dimension,
                                         np.array([DBU.lengths[j]]),
                                         np.array([DBU.centres[j]]))
                

                subtracted_DBU = subtracted_DBU.subtract_single_DBUs(DBU_2)

            if subtracted_DBU.no_of_boxes != 0:    
                final_DBU = final_DBU.append_disjoint_DBUs(subtracted_DBU)
            
        return final_DBU
    
    @staticmethod
    def has_zero(arr):
        return np.any(arr == 0)
    
    
    def subtract_single_DBUs(self, DBU):
        '''
        Given a DBU with one box and another DBU with just one box, subtract
        the self DBU from the other DBU. The stepsizes are borrowed from the DBU
        we subtract from.
        
        We do not return degenerate rectangles, which are those rectangles which have
        one dimension of the side length be 0

        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The DBU we wish to subtract from the original DBU.

        Returns:
        -----------------------------------------------------------------------
        subtracted_DBU : DisjointBoxUnion
                         The DBU we get after subtraction       
        '''
        centre1 = self.centres
        lengths1 = self.lengths
        centre2 = DBU.centres
        lengths2 = DBU.lengths
        stepsizes = self.stepsizes
        
        rectangle_list = op.subtract_rectangles(centre1[0], lengths1[0],
                                                centre2[0], lengths2[0])
        
        final_DBU = DisjointBoxUnion.empty_DBU(self.dimension)
        
        for rect_no in range(len(rectangle_list)):
            
            centres = np.expand_dims(rectangle_list[rect_no][0], axis=0)
            non_lengths = rectangle_list[rect_no][1]
            
            if not DisjointBoxUnion.has_zero(non_lengths):
                lengths = np.expand_dims(non_lengths, axis=0)
                
                final_DBU= final_DBU.append_disjoint_DBUs(DisjointBoxUnion(1,
                                                                       self.dimension,
                                                                       lengths,
                                                                       centres,
                                                                       stepsizes))
            
        return final_DBU
    
    def append_disjoint_DBUs(self, DBU):
        '''
        Append a DBU to the original DBU

        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The new DBU to append to the original DBU

        Returns:
        -----------------------------------------------------------------------
        appended_DBU : DisjointBoxUnion
                       The merged DBU

        '''
        new_box_number = DBU.no_of_boxes + self.no_of_boxes
        
        if np.sum(self.lengths) == 0:
            new_lengths = DBU.lengths
            new_centres = DBU.centres
            new_stepsizes = DBU.stepsizes
        else:
            new_lengths = np.concatenate((self.lengths, DBU.lengths), axis = 0)
            new_centres = np.concatenate((self.centres, DBU.centres), axis = 0)
            new_stepsizes = np.concatenate((self.stepsizes, DBU.stepsizes), axis =0)
        
        return DisjointBoxUnion(new_box_number, self.dimension, new_lengths, new_centres,
                                new_stepsizes)
    
    def DBU_union(self, DBU):
        '''
        Given a self DBU, take the union of this with the external DBU.
        
        # A1 sqcup A2 sqcup ... Ak union B = B sqcup A-B
        
        Returns:
        -----------------------------------------------------------------------
            new_DBU : DisjointBoxUnion
                      The new data structure which contains the union with the 
                      old DBU
        '''
        
        
        sub_DBU = self.subtract_DBUs(DBU)
        
        return DBU.append_disjoint_DBUs(sub_DBU)
        
    
    def get_bounds(self):
        '''
        Given a DBU get bounds on the various boxes present in the disjoint union.
        
        Returns:
        -----------------------------------------------------------------------
            bounds : List(2 * d) of length N
                     The bounds on the various boxes present in the DBU
        '''
        bounds = []
        
        for box_no in range(self.no_of_boxes):
            bounds.append([])
            for d in range(self.dimension):
                
                l_bound = self.centres[box_no, d] - self.lengths[box_no, d]/2
                r_bound = self.centres[box_no, d] + self.lengths[box_no, d]/2
                bounds[-1].append([l_bound, r_bound])
                
        self.bounds = bounds
        
        return bounds
    
    
    def get_total_bounds(self):
        '''
        Given a DBU get the maximum and the minimum bounds in the various dimensions for the distinct 
        boxes
        
        Returns:
        -----------------------------------------------------------------------
            total_bounds : np.array((dimension * 2))
                           The total bounds for the DBU
        Stores:
        -----------------------------------------------------------------------
            total_bounds : Same description as above
        '''
        total_bounds = np.array([[+np.inf, -np.inf] for d in range(self.dimension)])
        bounds = np.array(self.get_bounds())
        
        for d in range(self.dimension):
            for box_no in range(len(bounds)):
                
                if bounds[box_no,d,0] < total_bounds[d,0]:
                    total_bounds[d, 0] = bounds[box_no, d, 0]
                
                if bounds[box_no,d,1] > total_bounds[d,1]:
                    total_bounds[d,1] = bounds[box_no, d,1]
        
        self.total_bounds = total_bounds
        return total_bounds
    
    
    def get_end_points(self):
        '''
        Given a DBU, obtain the end points of the different boxes.

        Returns:
        -----------------------------------------------------------------------
            end_points : np.array(N * 2D * D)
                         The end points of the n'th box each of which has d coordinates

        '''
        end_points = []
        for box_no in range(self.no_of_boxes):
            
            center = self.centres[box_no]
            side_lengths = np.array(self.lengths[box_no])
        
            # Create an array of all possible combinations of -1 and 1 of length d
            offsets = np.array(np.meshgrid(*[[-0.5, 0.5]] * self.dimension)).T.reshape(-1, self.dimension)
        
            # Scale by the side lengths and add the center to get the endpoints
            vertices = center + offsets * side_lengths
            
            end_points.append(vertices)
        
        self.end_points = end_points
        
        return end_points
    
    
    @staticmethod
    def generate_k_tuples(array_list, k):
        '''
        Given a list of lists, return all possible k-tuples from the list of lists,
        and also return the indices. 

        Parameters:
        -----------------------------------------------------------------------
        lists : list[list]
                The list of lists we wish to sample from
        k : int
            The number of elements we want to look at in the tuples

        Returns:
        -----------------------------------------------------------------------
        results : tuple
                  The tuple which denotes the values and the indices we wish to
                  sample from 

        '''
        # Step 3: Generate all possible combinations of indices of size k
        index_combinations = list(itertools.combinations(range(len(array_list)), k))

        # Step 4: Generate all possible k-tuples for each combination
        all_k_tuples = []
        for indices in index_combinations:
            # Extract the sub-arrays corresponding to the current combination of indices
            sub_arrays = []
            for i in indices:
                cart_arr = itertools.product(array_list[i], repeat = 2)
                unequal_tuples = [tup for tup in cart_arr if tup[0]<tup[1]]
                sub_arrays.append(unequal_tuples)

            # Generate all possible k-tuples from the sub-arrays
            k_tuples = list(itertools.product(*sub_arrays))
            # Store the results
            all_k_tuples.append((indices, k_tuples))

    
        return all_k_tuples
    
    @staticmethod
    def tuples_to_array(tup):
        '''
        Given a tuple of tuples convert it into an array
        '''
        final_arr = []
        for i,elt in enumerate(tup):
            final_arr.append(np.array(elt))
        
        return np.array(final_arr)
    
    
    #Create an iterator method for conditions to speed up generate all conditions?
    
    def generate_all_conditions(self, max_complexity):
        '''
        For the given DBU, generate all the bounded conditions that is possible for
        a given complexity.
        
        Parameters:
        -----------------------------------------------------------------------
            max_complexity : int
                             Possible number of coordinates that can be modified in
                             the DBU.
            
        
        Returns:
        -----------------------------------------------------------------------
            conditions : list[constraint_conditions]
                         The list of bounded conditions that can be generated for a given
                         complexity for the given DBU.
        '''
        
        condition_list = []
        total_bounds = self.get_total_bounds()
        
        for box_no in range(self.no_of_boxes):
            cartesian_prod = []
            for dim in range(self.dimension):
                cartesian_prod.append(np.arange(self.get_bounds()[box_no][dim][0],
                                                self.get_bounds()[box_no][dim][1],
                                                self.stepsizes[box_no, dim]))
            
            for k in range(1, max_complexity+1):
                all_k_tuples = DisjointBoxUnion.generate_k_tuples(cartesian_prod, k)
                for indices, k_tuples in all_k_tuples:
                    for k_tuple in k_tuples:
                        k_array = DisjointBoxUnion.tuples_to_array(k_tuple)
                        constraint = cc.ConstraintConditions(dimension = self.dimension,
                                                             non_zero_indices = np.array(indices),
                                                             bounds = k_array,
                                                             state_bounds = total_bounds)
                        
                        #print(f'The nonzero indices are {indices}')
                        
                        condition_list.append(constraint)
            
        return condition_list
 
      
    def plot_2D(self, ax=None, show_plot=True, title=''):
        '''
        Plot a disjoint union of boxes in 2D
        '''
        if ax == None:
            fig,ax = plt.subplots()
        
        if self.bounds == None:
            self.get_bounds()
        #print(self.total_bounds)
        #print(type(self.total_bounds))
        
        for box_no in range(self.no_of_boxes):
            ax.scatter([self.centres[box_no, 0]], [self.centres[box_no, 1]],
                       color = 'red', marker = 'x')
            
            ax.hlines(y=self.centres[box_no, 1] + self.lengths[box_no, 1]/2, color='blue',
                       xmin=self.centres[box_no, 0] - self.lengths[box_no, 0]/2,
                       xmax=self.centres[box_no, 0] + self.lengths[box_no, 0]/2)
            
            ax.hlines(y=self.centres[box_no, 1] - self.lengths[box_no, 1]/2, color='blue',
                       xmin=self.centres[box_no, 0] - self.lengths[box_no, 0]/2,
                       xmax=self.centres[box_no, 0] + self.lengths[box_no, 0]/2)
            
            ax.vlines(x=self.centres[box_no, 0] + self.lengths[box_no, 0]/2, color='blue',
                       ymin = self.centres[box_no, 1] - self.lengths[box_no, 1]/2,
                       ymax = self.centres[box_no, 1] + self.lengths[box_no, 1]/2)
            
            ax.vlines(x=self.centres[box_no, 0] - self.lengths[box_no, 0]/2, color='blue',
                       ymin = self.centres[box_no, 1] - self.lengths[box_no, 1]/2,
                       ymax = self.centres[box_no, 1] + self.lengths[box_no, 1]/2)
        
        #ax.set_xlim(self.total_bounds[0,0]-1, self.total_bounds[1,0]+1)
        #ax.set_ylim(self.total_bounds[0,1]-1, self.total_bounds[1,1]+1)
        ax.set_title(title)
        if show_plot:
            plt.show()
            
        return ax
    
    
    def __str__(self):
        '''
        Print the various rectangles present in the DBU.
        
        '''
        dbu_string = ''
        for box_no in range(self.no_of_boxes):
            dbu_string += f'Box: {box_no}, Centre: {self.centres[box_no]}, Lengths: {self.lengths[box_no]}\n'
        
        return dbu_string    
    
    @staticmethod                                                              
    def empty_DBU(dimension):                                                  
        '''
        A static method that returns an empty DBU of a certain dimension     
        Parameters:                                                             
        -----------------------------------------------------------------------
        dimension : int
                    Dimension of the empty DBU

        Returns:
        -----------------------------------------------------------------------
        empty_DBU : type(DisjointBoxUnion)
                    An empty disjoint box union data structure

        '''
        return DisjointBoxUnion(0, dimension, np.array([[]]), np.array([[]]))

#%%
    
class DBUIterator:
    '''
    Create a DBU iterator for iterating through the points of the different boxes
    '''
    def __init__(self, DBU):
    
        self.DBU = DBU
        self.curr_box_no = 0
        self.point_index = np.zeros(self.DBU.dimension)
        self.first_point = True
        
    @staticmethod
    def count_points(a, b, d):
        '''
        Count the number of points in the list given by np.arange(a,b,d)

        Parameters:
        -----------------------------------------------------------------------
        a : float
            Lower Bound
        b : float
            Upper Bound
        d : float
            Difference

        Returns:
        -----------------------------------------------------------------------
        number_of_points : int
                           The number of points in the list

        '''
        return max(0, int(np.ceil(b-a) / d))
    
    @staticmethod
    def index_to_point(point_index, DBU, box_no):
        '''
        Given the set of indices for a certain box_no for a DBU return the corr
        point on the DBU

        Parameters:
        -----------------------------------------------------------------------
        point_index : np.array([int])
                      The indices of the point in the box

        Returns:
        -----------------------------------------------------------------------
        point : np.array[float]
                The point values on the box
        '''
        point = []
        for d in range(DBU.dimension):
            
            point.append(DBU.bounds[box_no][d][0] + (point_index[d]) * DBU.stepsizes[box_no, d])
            
        return point
    
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        for box_no in range(self.curr_box_no, self.DBU.no_of_boxes):
            
            self.lengths = [DBUIterator.count_points(
                                   self.DBU.bounds[box_no][d][0],
                                   self.DBU.bounds[box_no][d][1],
                                   self.DBU.stepsizes[box_no, d]) for
                                   d in range(self.DBU.dimension)]
            
            if self.first_point:
                self.first_point = False
                return DBUIterator.index_to_point(np.zeros(self.DBU.dimension), self.DBU, box_no)
                
            
            # Update the position for the next point
            for i in range(self.DBU.dimension - 1, -1, -1):

                if self.point_index[i] + 1 < self.lengths[i]:
                    self.point_index[i] += 1
                    point = DBUIterator.index_to_point(self.point_index ,self.DBU, box_no)
                    # We manually add this change so that we can return the true point
                    
                    return point
                else:
                    self.point_index[i] = 0
                    #print(f'Point index : {point_index}')
                    #print(f'Else condition reached and i is {i}')
                    if i == 0:
                        self.curr_box_no += 1
                        self.first_point = True
                
            self.point_index = np.zeros(self.DBU.dimension)
            
        if self.curr_box_no >= self.DBU.no_of_boxes:
            raise StopIteration

#%%
class ConditionsIterator:
    '''
    For a DBU and a given box we define an iterator to iterate through all the conditions
    that could possibly be generated with this method.
    '''
    
    def __init__(self, DBU, max_complexity):
        '''
        We generate and iterate through all the possible conditions

        Parameters:
        -----------------------------------------------------------------------
        DBU : DisjointBoxUnion
              The DBU over which we wish to generate all possible conditions
              
        curr_box_no : int
                 The box number in the DBU for which we wish to generate all possible constraint conditions
                 
        max_complexity : int
                         The maximum allowed complexity for a constraint condition

        Returns:
        -----------------------------------------------------------------------
        condition_iter class : ConditionIterator
                               The class which generates the iterations over all the conditions

        '''
        
        self.DBU = DBU
        self.max_complexity = max_complexity
        
        self.curr_box_no = 0
        self.complexity = 1
        self.possible_non_zero_indices = list(itertools.combinations(np.arange(self.DBU.dimension), self.complexity))
        self.curr_non_zero_index_position = len(self.possible_non_zero_indices)
        
        self.iter_dim = len(self.possible_non_zero_indices[-self.curr_non_zero_index_position:])
        self.bounds_per_box = DBU.get_bounds()
        self.curr_bounds = np.array(DBU.get_bounds()[0])
        self.total_bounds = DBU.get_total_bounds()
        
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        
        for box_no in range(self.curr_box_no, self.DBU.no_of_boxes):
            
            for complexity in range(self.complexity, self.max_complexity+1):
                
                for non_zero_indices in self.possible_non_zero_indices[-self.curr_non_zero_index_position:]:
                    non_zero_indices = np.array(non_zero_indices)
                    
                    for d in non_zero_indices[-self.iter_dim:]:
                        lengthstep = self.DBU.stepsizes[box_no, d]
                        #print(f'We check if {curr_bounds[d,0]} + {lengthstep} < {curr_bounds[d,1]}')
                        if self.curr_bounds[d, 0] + lengthstep < self.curr_bounds[d, 1]:
                            self.curr_bounds[d, 0] += lengthstep
                            #print(f'New left bounds {self.curr_bounds[d,0]}')
                            
                            #print(f'Complexity : {complexity}, non_zero_indices : {non_zero_indices}, and dimension {d}')
                            constraint = cc.ConstraintConditions(self.DBU.dimension,
                                                                 non_zero_indices,
                                                                 self.curr_bounds)
                            #print(f'The constraint is {constraint}')
                            #print(f'Box no: {box_no}, complexity: {complexity}, dimension: {d}, non_zero_indices {non_zero_indices}')
                            return constraint
                        
                        elif self.curr_bounds[d, 1] - lengthstep > np.array(self.bounds_per_box[box_no])[d,0]:
                            
                             self.curr_bounds[d, 1] -= lengthstep
                             #print(f'New right bounds {self.curr_bounds[d,1]}')
                             self.curr_bounds[d, 0] = np.array(self.bounds_per_box[box_no])[d, 0]
                            
                            #print(f'Complexity : {complexity}, non_zero_indices : {non_zero_indices}, and dimension {d}')
                             constraint = cc.ConstraintConditions(self.DBU.dimension,
                                                                 non_zero_indices,
                                                                 self.curr_bounds)
                            #print(f'The constraint is {constraint}')
                             #print(f'Box no: {box_no}, complexity: {complexity}, dimension: {d}, non_zero_indices {non_zero_indices}')
                             return constraint
                        
                        else:
                            self.curr_non_zero_index_position -= 1
                            
                            if self.curr_non_zero_index_position > 0:
                                self.curr_bounds = np.array(self.DBU.get_bounds()[self.curr_box_no])
                            
                            else:
                                self.iter_dim -= 1
                                if self.iter_dim > 0:
                                    self.curr_bounds = np.array(self.DBU.get_bounds()[self.curr_box_no])
                                    self.curr_non_zero_index_position = len(self.possible_non_zero_indices)
                                else:
                                    self.complexity += 1
                                    if self.complexity <= self.max_complexity:
                                        self.curr_bounds = np.array(self.DBU.get_bounds()[self.curr_box_no])
                                        self.possible_non_zero_indices = list(itertools.combinations(np.arange(self.DBU.dimension), self.complexity))
                                        self.curr_non_zero_index_position = len(self.possible_non_zero_indices)
                                        self.iter_dim = len(self.possible_non_zero_indices[-self.curr_non_zero_index_position:])
                                    
                                    else:
                                        self.curr_box_no += 1
                                        if self.curr_box_no < self.DBU.no_of_boxes:
                                            self.complexity = 1
                                            self.curr_bounds = np.array(self.DBU.get_bounds()[self.curr_box_no])
                                            self.possible_non_zero_indices = list(itertools.combinations(np.arange(self.DBU.dimension), self.complexity))
                                            self.curr_non_zero_index_position = len(self.possible_non_zero_indices)
                                            self.iter_dim = len(self.possible_non_zero_indices[-self.curr_non_zero_index_position:])
                                        
                                        else:
                                            print('Hurray! We reached the endpoint')
                                            raise StopIteration
                            
#%%
if __name__ == '__main__':
    dbu = DisjointBoxUnion(2, 2, np.array([[3,3], [2,2]]), np.array([[0,0], [-3,-3]]))
    dbu.get_bounds()[1]
    
    dbu = DisjointBoxUnion(2, 2, np.array([[3,3], [2,2]]), np.array([[0,0], [-3,-3]]))
    
    constraint_list = dbu.generate_all_conditions(2)
    fig,ax = plt.subplots()
    
    ax = dbu.plot_2D(ax, show_plot = False)
    for i,constraint in enumerate(constraint_list):
        constraint.plot_2D_constraints(ax, show_plot = True)
        fig, ax = plt.subplots()
        ax = dbu.plot_2D(ax, show_plot=False)
    
    for i,constraint in enumerate(constraint_list):
        print(f'The {i}th constraint is {constraint}')