# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 22:16:10 2023

@author: Kirill
"""
import numpy as np  		  	   		  		 			  		 			     			  	 
import random
from DTLearner import DTLearner

class RTLearner(DTLearner):
    """
    This is a RT Learner
    
    :param verbose: If “verbose” is True, your code can print out information for debugging.
                    If verbose = False your code should not generate ANY output.
                    When we test your code, verbose will be False.
    :type verbose: bool
    """
                		  	   		  		 			  		 			     			  	 
		  		 			  		 			     			  	   		  		 			  		 			     			  	 
    def __init__(self, leaf_size=1, verbose=False):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Constructor method  		  	   		  		 			  		 			     			  	 
        """  	
        super().__init__()	  	   		  		 			  		 			     			  	 
        self.leaf_size = leaf_size
        self.verbose = verbose  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    def author(self):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        :return: The GT username of the student  		  	   		  		 			  		 			     			  	 
        :rtype: str  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        return "kkalinin3"  # replace tb34 with your Georgia Tech username  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    def add_evidence(self, data_x, data_y):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Add training data to learner  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
        :param data_x: A set of feature values used to train the learner  		  	   		  		 			  		 			     			  	 
        :type data_x: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        :param data_y: The value we are attempting to predict given the X data  		  	   		  		 			  		 			     			  	 
        :type data_y: numpy.ndarray  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
  		  	   	
	  	#build tree here
        self.tree = self.build_tree(data_x, data_y) 
        
        if self.verbose == True:
            print("RTLearner")
            print("This is a tree")
            print(self.tree)
     
    def build_tree(self, data_x, data_y):
        
        if data_x.shape[0] == 1:
            return np.asarray([np.nan, np.mean(data_y), np.nan, np.nan])
        
        elif data_x.shape[0] <= self.leaf_size:
            return np.asarray([np.nan, np.mean(data_y), np.nan, np.nan])
        
        elif np.allclose(data_y, data_y[0]):    #same as == but for floating format
            return np.asarray([np.nan, data_y[0], np.nan, np.nan])
        else:
            select_feature = random.choice(range(0, data_x.shape[1]))
            p1, p2 = random.sample(range(data_x.shape[0]), 2)
            split_value = (data_x[p1, select_feature] + data_x[p2, select_feature]) / 2
           
            
            if split_value == max(data_x[:, select_feature]):
                return np.array([np.nan, np.mean(data_y), np.nan, np.nan])  
            
            left_mask = data_x[:, select_feature] <= split_value
            right_mask = data_x[:, select_feature] > split_value
            
            left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])
            right_tree = self.build_tree(data_x[right_mask], data_y[right_mask])
            
            if left_tree.ndim == 1:
                root = np.asarray([select_feature, split_value, 1, 2])
            else:
                root = np.asarray([select_feature, split_value, 1, left_tree.shape[0] + 1])
    
        return np.vstack((root, left_tree, right_tree))
  		  	   		  		 			  		 			     			  	 
if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    print("the secret clue is 'zzyzx'")  		  	   		  		 			  		 			     			  	 



