'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import pandas as pd

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainData = self.load_file(path + ".train.data.original")
        self.testData = self.load_file(path + ".test.data.original")
        #self.validData = self.load_file(path + ".valid.data")
        self.validData = self.testData
        
        
        self.num_users = self.trainData["UserID"].max()+1
        self.num_items = self.trainData["ItemID"].max()+1
        #self.trainData = self.trainData.sample(1000)
    def load_file(self, filename):        
        return pd.read_pickle(filename)
        