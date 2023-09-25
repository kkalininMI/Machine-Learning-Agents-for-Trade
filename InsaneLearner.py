import BagLearner as bl
import LinRegLearner as lrl	   	  			  	 		  		  		    	 		 		   		 		  	 		  		  		    	 		 		   		 		  
class InsaneLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False):
        self.learners = [bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False)] * 20
    def author(self):
        return 'kkalinin3'
    def add_evidence(self, data_x, data_y):
        return [learner.add_evidence(data_x, data_y) for learner in self.learners]	  		  		    	 		 		   		 	
    def query(self,points):
        res = [learner.query(points)  for learner in self.learners]
        return sum(res) / len(res)	