""""""  		  	   		  		 			  		 			     			  	 
"""  		  	   		  		 			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
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
  		  	   		  		 			  		 			     			  	 
import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il




### Functions

def calculate_rmse(train_y, pred_y):
    return math.sqrt(((pred_y - train_y) ** 2).sum() / train_y.shape[0])


def experiment_1(train_x, train_y, test_x, test_y):
    max_leaf = 50
    insample_list = []
    outsample_list = []

    for leaf_size in range(1, max_leaf + 1):
        
        learner = dt.DTLearner(leaf_size = leaf_size)
        
        learner.add_evidence(train_x, train_y)

        insample_list.append(
            calculate_rmse(
                train_y, learner.query(train_x)))
        
        outsample_list.append(
            calculate_rmse(
                test_y, learner.query(test_x)))
        
    ruler = range(0, max_leaf)
    
    plt.plot(ruler, insample_list, label = "in-sample")
    plt.plot(ruler, outsample_list, label = "out-sample")
    plt.title("Figure 1. Overfitting in DTLearner")
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.xticks(np.arange(0, max_leaf + 10, step = 10))
    plt.grid()
    plt.legend()
    plt.savefig("figure1.png")
    plt.clf()


def experiment_2(train_x, train_y, test_x, test_y):
       max_leaf = 50
       bag_size = 30
       insample_list = []
       outsample_list = []
       
       for leaf_size in range(1, max_leaf + 1):
           
           learner = bl.BagLearner(learner = dt.DTLearner,
                                kwargs = {"leaf_size": leaf_size}, 
                                bags = bag_size)
        
           learner.add_evidence(train_x, train_y)

           insample_list.append(
               calculate_rmse(
                   train_y, learner.query(train_x)))
           
           outsample_list.append(
               calculate_rmse(
                   test_y, learner.query(test_x)))     
           
       ruler = range(0, max_leaf)
       
       plt.plot(ruler, insample_list, label = "in-sample")
       plt.plot(ruler, outsample_list, label = "out-sample")
       plt.title("Figure 2. Overfitting in Bagging (DTLearner), bag size 30")
       plt.xlabel("Leaf Size")
       plt.ylabel("RMSE")
       plt.xticks(np.arange(0, max_leaf + 10, step = 10))
       plt.grid()
       plt.legend()
       plt.savefig("figure2.png")
       plt.clf()


def experiment_3_1(train_x, train_y, test_x, test_y):
    max_leaf = 50

    out_sample_dt = []
    out_sample_rt = []

    def r_squared(y_obs, y_pred):
        tss = np.sum((y_obs - np.mean(y_obs)) ** 2)
        rss = np.sum((y_obs - y_pred) ** 2)
        r_squared = 1 - (rss / tss)
        return r_squared

    for leaf_size in range(1, max_leaf + 1):
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)
        out_sample_dt.append(
            r_squared(np.asarray(test_y),
                      np.asarray(learner.query(test_x))))

        learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)
        out_sample_rt.append(
            r_squared(np.asarray(test_y),
                      np.asarray(learner.query(test_x))))

    ruler = range(1, max_leaf + 1)

    plt.plot(ruler, out_sample_dt, label="DTLearner")
    plt.plot(ruler, out_sample_rt, label="RTLearner")
    plt.title("Figure 3.1 Comparing DTLearner and RTLearner (R-Squared)")
    plt.xlabel("Leaf size")
    plt.ylabel("R-Squared")
    plt.xticks(np.arange(0, max_leaf + 10, step=10))
    plt.grid()
    plt.legend()
    plt.savefig("figure3_1.png")
    plt.clf()


def experiment_3_2(train_x, train_y, test_x, test_y):
       max_leaf = 50
       
       out_sample_dt = []
       out_sample_rt = []
           
       for leaf_size in range(1, max_leaf + 1):
           
           learner = dt.DTLearner(leaf_size = leaf_size, verbose = False)
           learner.add_evidence(train_x, train_y)
           out_sample_dt.append(
               calculate_rmse(
                   test_y, learner.query(test_x)))
           
           learner = rt.RTLearner(leaf_size = leaf_size, verbose = False)
           learner.add_evidence(train_x, train_y)
           out_sample_rt.append(
               calculate_rmse(
                   test_y, learner.query(test_x)))
           
       ruler = range(1, max_leaf + 1)
       
       plt.plot(ruler, out_sample_dt, label = "DTLearner")
       plt.plot(ruler, out_sample_rt, label = "RTLearner")
       plt.title("Figure 3.2 Comparing DTLearner and RTLearner (MAE)")
       plt.xlabel("Leaf size")
       plt.ylabel("MAE")
       plt.xticks(np.arange(0, max_leaf + 10, step = 10))
       plt.grid()
       plt.legend()
       plt.savefig("figure3_2.png")
       plt.clf()    

def experiment_3_3(train_x, train_y, test_x, test_y):
       max_leaf = 50
       out_sample_dt = []
       out_sample_rt = []
               
       for leaf_size in range(1, max_leaf + 1):
           learner = dt.DTLearner(leaf_size = leaf_size, verbose = False)
           
           start = time.time()
           learner.add_evidence(train_x, train_y)
           end = time.time()
           out_sample_dt.append(end - start)
           
           learner = rt.RTLearner(leaf_size = leaf_size, verbose = False)
           start = time.time()
           learner.add_evidence(train_x, train_y)
           end = time.time()
           out_sample_rt.append(end - start)
           
       ruler = range(1, max_leaf + 1)
       
       plt.plot(ruler, out_sample_dt, label = "DTLearner")
       plt.plot(ruler, out_sample_rt, label = "RTLearner")
       plt.title("Figure 3.3 Comparing DTLearner and RTLearner (Training Time)")
       plt.xlabel("Leaf size")
       plt.ylabel("Training Time")
       plt.xticks(np.arange(0, max_leaf + 10, step = 10))
       plt.grid()
       plt.legend()
       plt.savefig("figure3_3.png")
       plt.clf()
       

	  	   		  		 			  		 			     			  	 
if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    if len(sys.argv) != 2:  		  	   		  		 			  		 			     			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		  		 			  		 			     			  	 
        sys.exit(1)
    

   #inf = open("Data/Istanbul.csv")  	  	   		  		 			  		 			     			  	 
    inf = open(sys.argv[1])
    data = np.array([list(map(str, s.strip().split(","))) for s in inf.readlines()])
    
    if sys.argv[1] == "Data/Istanbul.csv":
        data = data[1:,1:]	
    
    data = np.array(data, dtype=np.float32)
			  		 			     			  	   		  	   		  		 			  		 			     			  	 
    # compute how much of the data is training and testing  		  	   		  		 			  		 			     			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		  		 			  		 			     			  	 
    test_rows = data.shape[0] - train_rows  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    # separate out training and testing data  		  	   		  		 			  		 			     			  	 
    train_x = data[:train_rows, 0:-1]  		  	   		  		 			  		 			     			  	 
    train_y = data[:train_rows, -1]  		  	   		  		 			  		 			     			  	 
    test_x = data[train_rows:, 0:-1]  		  	   		  		 			  		 			     			  	 
    test_y = data[train_rows:, -1]  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    print(f"{test_x.shape}")  		  	   		  		 			  		 			     			  	 
    print(f"{test_y.shape}")  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    # create a learner and train it  		  	   		  		 			  		 			     			  	 
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner  		  	   		  		 			  		 			     			  	 
    learner.add_evidence(train_x, train_y)  # train it  		  	   		  		 			  		 			     			  	 
    print(learner.author())  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    # evaluate in sample  		  	   		  		 			  		 			     			  	 
    pred_y = learner.query(train_x)  # get the predictions  		  	   		  		 			  		 			     			  	 
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		  		 			  		 			     			  	 
    print()  		  	   		  		 			  		 			     			  	 
    print("In sample results")  		  	   		  		 			  		 			     			  	 
    print(f"RMSE: {rmse}")  		  	   		  		 			  		 			     			  	 
    c = np.corrcoef(pred_y, y=train_y)  		  	   		  		 			  		 			     			  	 
    print(f"corr: {c[0,1]}")  		  	   		  		 			  		 			     			  	 
  		  	   		  		 			  		 			     			  	 
    # evaluate out of sample  		  	   		  		 			  		 			     			  	 
    pred_y = learner.query(test_x)  # get the predictions  		  	   		  		 			  		 			     			  	 
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		  		 			  		 			     			  	 
    print()  		  	   		  		 			  		 			     			  	 
    print("Out of sample results")  		  	   		  		 			  		 			     			  	 
    print(f"RMSE: {rmse}")  		  	   		  		 			  		 			     			  	 
    c = np.corrcoef(pred_y, y=test_y)  		  	   		  		 			  		 			     			  	 
    print(f"corr: {c[0,1]}")  	
    
    #Experiments
    experiment_1(train_x, train_y, test_x, test_y)
    experiment_2(train_x, train_y, test_x, test_y)
    experiment_3_1(train_x, train_y, test_x, test_y)
    experiment_3_2(train_x, train_y, test_x, test_y)
    experiment_3_3(train_x, train_y, test_x, test_y)