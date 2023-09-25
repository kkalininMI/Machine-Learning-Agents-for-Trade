""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  		 			  		 			     			  	 
All Rights Reserved  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			     			  	 
or edited.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			     			  	 
GT honor code violation.  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
import numpy as np  		  	   		  		 			  		 			     			  	 
 
 
       #tree_method
        
#        data_y = data[:,-1]
#        data_x = data[:,:-1]
 
		  	   		  		 			  		 			     			  	  		  	   		  		 			  		 			     			  	 
class DTLearner(object):  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    This is a DT Learner.			  	 
  		  	   		  		 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 			  		 			     			  	 
    :type verbose: bool  		  	   		  		 			  		 			     			  	 
    """  		  	   		  		 			  		 			     			  	 
    def __init__(self, leaf_size=1, verbose=False):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Constructor method  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
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
            print("DTLearner")
            print("This is a tree")
            print(self.tree)
         
    def build_tree(self, data_x, data_y):
    
        if data_x.shape[0] == 1:
            split_value = np.mean(data_y)
            return np.asarray([np.nan, np.mean(data_y), np.nan, np.nan])
        
        elif data_x.shape[0] <= self.leaf_size:
            split_value = np.mean(data_y)
            return np.asarray([np.nan, np.mean(data_y), np.nan, np.nan])
        
        elif np.allclose(data_y, data_y[0]):    #same as == but for floating format
            return np.asarray([np.nan, data_y[0], np.nan, np.nan])
        else:
        
            select_feature = self.determine_feature (data_x, data_y) 
            split_value = np.median(data_x[:, select_feature])
            
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


    def determine_feature (self, data_x, data_y): 
        max_cor, max_index = -1, 0
        
        for i in range(data_x.shape[1]):
            
            if np.std(data_x[:,i]) > 0:
                corr = np.corrcoef(data_x[:,i], data_y)[0,1]
            else:
                corr = 0

            if corr > max_cor:
                max_cor = corr
                max_index = i
                
        return max_index
    
    
    def query(self, dpoints):  		   	  			  	 		  		  		    	 		 		   		 		  
            res = []
            for dpoint in dpoints:
                res.append(self.tree_prediction(dpoint))
            return np.asarray(res)

    def tree_prediction(self, dpoint):
        row = 0 
        if self.tree.ndim < 2:
            return np.asarray(np.nan)
    
        while ~np.isnan(self.tree[row, 0]):
            value = dpoint[int(self.tree[row, 0])]
          
            if value <= self.tree[row][1]:
                row = row + int(self.tree[row, 2])
            else:
                row = row + int(self.tree[row, 3])
            
            result = self.tree[row, 1]
        return result
 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    print("the secret clue is 'zzyzx'")  		  	   		  		 			  		 			     			  	 
