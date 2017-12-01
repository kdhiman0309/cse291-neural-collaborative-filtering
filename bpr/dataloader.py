'''
Created on Apr 12, 2016

@author: hexiangnan
'''
import pandas as pd
from Dataset import Dataset
import pickle
from collections import defaultdict

class DatasetBPR():
    def __init__(self,datapath, prep_data=False):
        """
        Each line of .rating file is: userId(starts from 0), itemId, ratingScore, time
        Each element of train is the [[item1, time1], [item2, time2] of the user, sorted by time
        Each element of test is the [user, item, time] interaction, sorted by time
        """
        if prep_data:
            d = Dataset(datapath, prep_data=False)
            print("data loaded")
            num_user = d.num_users
            num_item = d.num_items
            
            train = defaultdict(list)
            d.trainData = d.trainData.sort_values("Timestamp")
            d.testData = d.testData.sort_values("Timestamp")
            
            for index, row in d.trainData.iterrows():
                user = row["UserID"]
                item = row["ItemID"]
                train[user].append(item)
            
            
            self.train = train
            self.testData = d.testData
            self.num_user = num_user
            self.num_item = num_item
            
            self.pickleit(self.train, datapath+".bpr.train.data")
            self.pickleit(self.testData, datapath+".bpr.test.data")
            self.pickleit([self.num_user, self.num_item], datapath+".bpr.config.data")
            
        else:
            self.train = self.loadPickle(datapath+".bpr.train.data")
            self.testData = self.loadPickle(datapath+".bpr.test.data")
            
            _t = self.loadPickle(datapath+".bpr.config.data")
            self.num_user, self.num_item = _t[0], _t[1]
            
    def pickleit(self, o, path):
        
        with open(path, 'wb') as f:
            pickle.dump(o, f)
            
    def loadPickle(self, path):
        with open(path, 'rb') as f:
            o = pickle.load(f)
        return o

            
        