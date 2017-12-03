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
        
class ItemFeatures():
    def __init__(self):
        self.descp = None
        self.genre = None
        self.year = None
    
class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path, prep_data=False,count_per_user_test=2,count_per_user_validation=0,num_negatives_train=4,num_threads=1,gensim_dim=10):
        '''
        Constructor
        '''
        self.num_threads=num_threads
        self.item_features = self.load_file(path + ".itemfeatures_gensim_%d.pkl"%gensim_dim)
        self.item_list = list(self.item_features.index)
            
        #self.trainData = self.load_file(path + ".train.data")
        self.testData = self.load_file(path + ".test.data")
        self.testColdStart = self.load_file(path + ".testColdStart.data")
        self.testColdStartPseudo = self.load_file(path + ".testColdStartPseudo.data")
        ##self.validData = self.load_file(path + ".valid.data")
        #self.validData = self.testData
        #self.test_data = self.loadPickle(path+".test_data") 
        self.train_data = self.loadPickle(path+".train_data")
            
        if prep_data:
            self.train_data_item_feat = ItemFeatures()
            self.add_item_features_info()
            self.pickleit(self.train_data_item_feat,path+".train_data_item_feat_%d.pkl"%gensim_dim)
        else:
            self.train_data_item_feat = self.loadPickle(path+".train_data_item_feat_%d.pkl"%gensim_dim)
        
        
        self.num_users = 7000#self.trainData["UserID"].max()+1
        self.num_items = len(self.item_features)
        #self.trainData = self.trainData.sample(1000)
        
        
    def load_file(self, filename):        
        return pd.read_pickle(filename)
    
          

    def get_item_feature_bulk(self,itemids):
        curr_row = self.item_features.loc[itemids]
        return  np.array(curr_row["Storyline_gensim"].tolist()), np.array(curr_row["Genre"].tolist()), np.array(curr_row["Year"].tolist())
    
    def get_item_feature(self,itemid):
      curr_row = self.item_features.loc[itemid]
      return curr_row["Storyline_gensim"],curr_row["Genre"],curr_row["Year"]
     
    def _one_train_data(self, index):
        user_input, item_input, labels = [],[],[]
        
        row = self.trainData.loc[index]
        # positive instance
        u = row["UserID"]
        i = row["ItemID"]
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        
        negatives = row["Negatives"]
        for _i in range(len(negatives)):
            neg_item_ID = negatives[_i]
            user_input.append(u)
            item_input.append(neg_item_ID)
            labels.append(0)
        
        return user_input, item_input, labels
    
    def add_item_features_info(self):
        t1 = time()
        genre=[None]*len(self.train_data.userids)
        descp = [None]*len(self.train_data.userids)
        year = [None]*len(self.train_data.userids)
        item_ids = self.train_data.itemids
        for i in range(len(self.train_data.userids)):
            d,g,y = self.get_item_feature(item_ids[i])
            descp[i] = d
            genre[i] = g
            year[i] = y
            
        self.train_data_item_feat.genre = np.array(genre)
        self.train_data_item_feat.descp = np.array(descp)
        self.train_data_item_feat.year = np.array(year)
        print("add_item_features_info %f",time()-t1)
        
    def generator_train_data(self,batch_size):
        while(True):
            user_ids = self.train_data.userids
            item_ids = self.train_data.itemids
            _labels = self.train_data.labels
            user_ids,item_ids,_labels = shuffle(user_ids,item_ids,_labels)
        
            i=0
            while(i <len(user_ids)):
                users, items, decs, genre, year, labels = [],[],[],[],[],[]
                    
                for j in range(batch_size):
                    if i>=len(user_ids):
                        break
                    d,g,y = self.get_item_feature(item_ids[i])
                    
                    users.append(user_ids[i])
                    items.append(item_ids[i])
                    decs.append(d)
                    genre.append(g)
                    year.append(y)
                    labels.append(_labels[i])
                    i+=1
                users, items, decs, genre, year, labels = \
                    np.array(users),np.array(items),np.array(decs),np.array(genre),np.array(year),np.array(labels), 
                yield([users, items, decs, genre, year],labels)
                
    
    def generator_test_data(self):
        user_ids = self.test_data.userids
        item_ids = self.test_data.itemids
        
        for i in range(len(user_ids)):
            d,g,y = self.get_item_feature(item_ids[i])
            yield [user_ids[i],item_ids[i],d,g,y]
    def save(self, path):
        self.trainData.to_pickle(path+".train.data")
        self.testColdStart.to_pickle(path+".testColdStart.data")
        self.testColdStartPseudo.to_pickle(path+".testColdStartPseudo.data")
        
        self.testData.to_pickle(path+".test.data")
        #self.validData.to_pickle(path+".valid.data")
        
        self.pickleit(self.train_data, path+".train_data")
        
    def pickleit(self, o, path):
        
        with open(path, 'wb') as f:
            pickle.dump(o, f)
            
    def loadPickle(self, path):
        with open(path, 'rb') as f:
            o = pickle.load(f)
        return o
