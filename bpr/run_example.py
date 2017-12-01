'''
Created on Apr 15, 2016

An Example to run the MFbpr

@author: hexiangnan
'''
from MFbpr import MFbpr
import multiprocessing as mp
from dataloader import DatasetBPR

def train():
    
    # Load data
    datapath = "../data/movielens20M"
    #train, test, num_user, num_item, num_ratings = LoadRatingFile_HoldKOut(dataset, splitter, hold_k_out)
    d = DatasetBPR(datapath, prep_data=False)
    train, test, num_user, num_item = d.train, d.testData, d.num_user, d.num_item
    
    print("Load data (%s) done." %(datapath))
    #print("#users: %d, #items: %d, #ratings: %d" %(num_user, num_item, num_ratings))
    
    # MFbpr parameters
    factors = 8
    learning_rate = 0.01
    reg = 0.01
    init_mean = 0
    init_stdev = 0.01
    maxIter = 100
    num_thread = 1
    
    # Run model
    bpr = MFbpr(train, test, num_user, num_item, 
                factors, learning_rate, reg, init_mean, init_stdev)
    bpr.build_model(maxIter, num_thread, batch_size=32)
    
train()