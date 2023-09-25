# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:30:08 2023

@author: Kirill
"""

import numpy as np  		  	   		  		 			  		 			     			  	 

class BagLearner(object):
    """
    This is a Bag Learner
    
    :param verbose: If “verbose” is True, your code can print out information for debugging.
                    If verbose = False your code should not generate ANY output.
                    When we test your code, verbose will be False.
    :type verbose: bool
    """
             

#import BagLearner as bl  
#learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)  
#learner.add_evidence(Xtrain, Ytrain)  
#Y = learner.query(Xtest) 
   		  	   		  		 			  		  		 			     			  	 
    def __init__(self, learner, kwargs, bags, boost = False, verbose = False):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Constructor method  		  	   		  		 			  		 			     			  	 
        """  	
        self.boost = boost
        self.verbose = verbose  
        self.learner_ensemble = [learner(**kwargs) for i in range(bags)] 

         		 			     			  		  	   		  		 			  		 			     			  	 
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
        
        for learner in self.learner_ensemble:
            index = np.random.choice(range(0, data_x.shape[0]), data_x.shape[0], replace = True)
            bag_x = data_x[index]
            bag_y = data_y[index]
            learner.add_evidence(bag_x, bag_y)
            
    def query(self, points):
        result = [learner.query(points) for learner in self.learner_ensemble]    
		   	  		  	 		  		  		    	 			   		 		  
        #for learner in self.learners:
        #    out.append(learner.query(points))
        return sum(result) / len(result)	



  		  	   		  		 			  		 			     			  	 
if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    print("the secret clue is 'zzyzx'")  		  	   		  		 			  		 			     			  	 



