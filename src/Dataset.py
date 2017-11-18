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
        self.trainData = self.load_file(path + ".train.data")
        self.testData = self.load_file(path + ".test.data")
        
        
        self.num_users = self.trainData["UserID"].max() 
        self.num_items = self.trainData["ItemID"].max()
        
    def load_file(self, filename):        
        return pd.read_csv(filename)        