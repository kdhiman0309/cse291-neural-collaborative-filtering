'''
Created on Aug 8, 2016
Processing datasets. 


'''
import scipy.sparse as sp
import numpy as np
import pandas as pd
from collections import defaultdict
import copy
from sklearn.utils import shuffle
from time import time
import pickle
import multiprocessing
import random

class ModelData():
    def __init__(self):
        self.userids = None
        self.itemids = None
        self.labels = None
        self.descp = None
        self.genre = None
        self.year = None
        self.gtitem = None
        
        self.pw_userids = None
        self.pw_posids = None
        self.pw_negids = None
        
class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path, prep_data=False, count_per_user_test=2,count_per_user_validation=0,num_negatives_train=4,num_threads=1):
        '''
        Constructor
        '''
        print("dataset")
        self.num_threads=num_threads
        self.item_features = self.load_file(path + ".itemfeatures.pkl")
        self.item_list = list(self.item_features.index)
        
        # prep-data must be from src/Dataset.py
        
        #self.trainData = self.load_file(path + ".train.data")
        self.testData = self.load_file(path + ".test.data")
        self.testColdStart = self.load_file(path + ".testColdStart.data")
        self.testColdStartPseudo = self.load_file(path + ".testColdStartPseudo.data")
        ##self.validData = self.load_file(path + ".valid.data")
        #self.validData = self.testData
        #self.test_data = self.loadPickle(path+".test_data")    
        self.train_data = self.loadPickle(path+".train_data")
        
        self.num_users = 10000#self.trainData["UserID"].max()+1
        self.num_items = len(self.item_features)
        #self.trainData = self.trainData.sample(10000)
        
        
    def load_file(self, filename):        
        return pd.read_pickle(filename)
    
    def pickleit(self, o, path):
        
        with open(path, 'wb') as f:
            pickle.dump(o, f)
            
    def loadPickle(self, path):
        with open(path, 'rb') as f:
            o = pickle.load(f)
        return o
