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

    def __init__(self, path, prep_data=False,count_per_user_test=2,count_per_user_validation=0,num_negatives_train=4,num_threads=1):
        '''
        Constructor
        '''
        print("dataset")
        self.num_threads=num_threads
        self.item_features = self.load_file(path + ".itemfeatures.pkl")
        if prep_data:
            self.item_list = list(self.item_features.index)
            self.trainData = self.load_file(path+".ratings.data.pkl")
            #self.trainData = self.trainData.sample(frac=0.01)
            self.split_cold_start_items()
            self.split_train_test(count_per_user_test=count_per_user_test,count_per_user_validation=count_per_user_validation)
            self.negative_sampling(num_negatives_train=num_negatives_train)
            self.gen_train_data()
            #self.gen_test_data()
            self.save(path)
            
        else:
            #self.trainData = self.load_file(path + ".train.data")
            self.testData = self.load_file(path + ".test.data")
            self.testColdStart = self.load_file(path + ".testColdStart.data")
            self.testColdStartPseudo = self.load_file(path + ".testColdStartPseudo.data")
           # self.validData = self.load_file(path + ".valid.data")
            #self.validData = self.testData
            #self.test_data = self.loadPickle(path+".test_data")    
            self.train_data = self.loadPickle(path+".train_data")
            
        self.num_users = 7000#self.trainData["UserID"].max()+1
        self.num_items = len(self.item_features)
        #self.trainData = self.trainData.sample(1000)
        
        
    def load_file(self, filename):        
        return pd.read_pickle(filename)
    
    def split_cold_start_items(self, frac=0.1, num_pairs_in_train=10):
        items = self.item_features.ItemID.unique()
        cold_set_items = random.sample(set(items), int(len(items)*frac))
        cold_set_items = shuffle(cold_set_items)
        cold_set_items_pseudo = cold_set_items[:int(len(cold_set_items)/2)]
        cold_set_items = cold_set_items[int(len(cold_set_items)/2):]
        cold_set_items = set(cold_set_items)
        cold_set_items_pseudo = set(cold_set_items_pseudo)
        item_counts = defaultdict(int)
        
        def map_it(x):
            if x in cold_set_items_pseudo: 
                if item_counts[x] < num_pairs_in_train:
                    item_counts[x]+=1
                    return False
                else:
                    return True
            else:
                return False
        
        self.train_data = shuffle(self.trainData)
        
        self.testColdStartPseudo = self.trainData[self.trainData.ItemID.apply(lambda x: map_it(x))]
        self.testColdStart = self.trainData[self.trainData.ItemID.apply(lambda x: x in cold_set_items)]
        
        self.trainData = self.trainData.loc[list(set(self.trainData.index) - set(self.testColdStart.index) - set(self.testColdStartPseudo.index))]
    
    def split_train_test(self,count_per_user_test=1,count_per_user_validation=0):
        df = self.trainData
        df = df.sort_values("Timestamp", ascending=False)
        #df = shuffle(df)
        
        user_counts = defaultdict(int)
        rows = []
        rows_valid = []
        for index, row in df.iterrows():
            if user_counts[row["UserID"]] < count_per_user_test:
                rows += [index]
                user_counts[row["UserID"]] +=1
            elif user_counts[row["UserID"]] < count_per_user_test+count_per_user_validation:
                rows_valid += [index]
                user_counts[row["UserID"]] +=1
            
        df_test_ratings = df.loc[rows]
        df_validation_ratings = df.loc[rows_valid]
        df_train_ratings = df.loc[list(set(df.index)-set(rows)-set(rows_valid))]
    
        self.trainData = df_train_ratings
        self.testData  = df_test_ratings
        self.validData = df_validation_ratings
    
    def negative_sampling(self, num_negatives_train=4):
        user_item_pairs = defaultdict(set)
        
        for _data in [self.testData, self.testColdStart, self.testColdStartPseudo, self.trainData]:
            for x in _data[["UserID", "ItemID"]].values:
                user_item_pairs[x[0]].add(x[1])
            
        
        num_items = len(self.item_list)
        user_item_pairs2 = copy.deepcopy(user_item_pairs)
        
        def get_negs(u):
            negs = []
            for _i in range(num_negs_per_positive):
                # generate negatives and add to dict
                j = np.random.randint(num_items)
                j = self.item_list[j]    
                while (u,j) in user_item_pairs[u]:
                     j = np.random.randint(num_items)
                     j = self.item_list[j]
                    #print(".")
                    #assert(j in self.item_list)
                 
                negs += [j]
                #user_item_pairs[u].add(j)
                user_item_pairs2[u].add(j)
            return negs
        
        num_negs_per_positive = 99
        self.testData["Negatives"] = self.testData["UserID"].apply(lambda x: get_negs(x))
        #user_item_pairs = user_item_pairs2
        #user_item_pairs2 = copy.deepcopy(user_item_pairs)
        
        num_negs_per_positive = 99
        self.testColdStart["Negatives"] = self.testColdStart["UserID"].apply(lambda x: get_negs(x))
        #user_item_pairs = user_item_pairs2
        #user_item_pairs2 = copy.deepcopy(user_item_pairs)
        
        num_negs_per_positive = 99
        self.testColdStartPseudo["Negatives"] = self.testColdStartPseudo["UserID"].apply(lambda x: get_negs(x))
        user_item_pairs = user_item_pairs2
        user_item_pairs2 = copy.deepcopy(user_item_pairs)
        
        #num_negs_per_positive = 99
        #self.validData["Negatives"] = self.validData["UserID"].apply(lambda x: get_negs(x))
        #user_item_pairs = user_item_pairs2
        #user_item_pairs2 = copy.deepcopy(user_item_pairs)
    
        num_negs_per_positive = num_negatives_train
        self.trainData["Negatives"] = self.trainData["UserID"].apply(lambda x: get_negs(x))
          

    def get_item_feature(self,itemid):
      curr_row = self.item_features.iloc[itemid]
      return curr_row["Description"],curr_row["Genre"],curr_row["Year"]
     
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
    
    def gen_train_data(self):
        t1 = time()
        user_input, item_input, labels = [],[],[]
        user_pw, pos_input, neg_input = [], [], []
        if self.num_threads>1:
            pool = multiprocessing.Pool(processes=self.num_threads)
            res = pool.map(self._one_train_data, self.trainData.index)
            pool.close()
            pool.join()
            for _t in res:
                user_input += _t[0]
                item_input += _t[1]
                labels     += _t[2]
            
        else:
            for index in self.trainData.index:
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
                    
                    user_pw.append(u)
                    pos_input.append(i)
                    neg_input.append(neg_item_ID)
                
        user_input, item_input, labels = shuffle(user_input, item_input, labels)
        user_pw, pos_input, neg_input = shuffle(user_pw, pos_input, neg_input)
        
        m = ModelData()
        
        m.userids = np.array(user_input)
        m.itemids = np.array(item_input)
        m.labels = np.array(labels)
        
        m.pw_userids = np.array(user_pw)
        m.pw_posids = np.array(pos_input)
        m.pw_negids = np.array(neg_input)
        
        self.train_data = m
        print("gen_train_data [%.1f s]"%(time()-t1))
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
