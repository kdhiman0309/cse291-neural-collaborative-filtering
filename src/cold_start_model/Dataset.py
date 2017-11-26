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

class ModelData():
    def __init__(self):
        self.userids = None
        self.itemids = None
        self.labels = None
        self.descp = None
        self.genre = None
        self.year = None
        self.gtitem = None
        
class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path, prep_data=False,count_per_user_test=1,count_per_user_validation=0,num_negatives_train=4,num_threads=1):
        '''
        Constructor
        '''
        self.num_threads=num_threads
        self.item_features = self.load_file(path + ".itemfeatures.pkl")
        if prep_data:
            self.item_list = list(self.item_features.index)
            self.trainData = self.load_file(path+".ratings.data.pkl")
            self.trainData = self.trainData.sample(frac=0.01)
            
            self.split_train_test(count_per_user_test=count_per_user_test,count_per_user_validation=count_per_user_validation)
            self.negative_sampling(num_negatives_train=num_negatives_train)
            self.gen_train_data()
            self.gen_test_data()
            self.save(path)
            
        else:
            self.trainData = self.load_file(path + ".train.data")
            self.testData = self.load_file(path + ".test.data")
            #self.validData = self.load_file(path + ".valid.data")
            self.validData = self.testData
            self.test_data = self.loadPickle(path+".test_data")    
            self.train_data = self.loadPickle(path+".train_data")
            
        self.num_users = 14182#self.trainData["UserID"].max()+1
        self.num_items = len(self.item_features)
        #self.trainData = self.trainData.sample(1000)
        
        
    def load_file(self, filename):        
        return pd.read_pickle(filename)
    
    def split_train_test(self,count_per_user_test=1,count_per_user_validation=0):
        df = self.trainData
        df = df.sort_values("Timestamp", ascending=False)
       
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
        
        for x in self.testData[["UserID", "ItemID"]].values:
            user_item_pairs[x[0]].add(x[1])
            
        for x in self.trainData[["UserID", "ItemID"]].values:
            user_item_pairs[x[0]].add(x[1])
            
        for x in self.validData[["UserID", "ItemID"]].values:
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
        user_item_pairs = user_item_pairs2
        user_item_pairs2 = copy.deepcopy(user_item_pairs)
        
        
        num_negs_per_positive = 99
        self.validData["Negatives"] = self.validData["UserID"].apply(lambda x: get_negs(x))
        user_item_pairs = user_item_pairs2
        user_item_pairs2 = copy.deepcopy(user_item_pairs)
    
        num_negs_per_positive = num_negatives_train
        self.trainData["Negatives"] = self.trainData["UserID"].apply(lambda x: get_negs(x))
        
        

    def get_item_feature(self,itemid):
      curr_row = self.item_features.iloc[itemid]
      return curr_row["Description"],curr_row["Genre"],curr_row["Year"]
  
    def _one_test_data(self,_i):
        index = self.testData.index[_i]
        row = self.testData.loc[index]
    #for index, row in self.testData.iterrows():
        items = row["Negatives"]#_testNegatives[idx]
        u = row["UserID"]
        gtItem = row["ItemID"]
        items.append(gtItem)
        # Get prediction scores
        users = np.full(len(items), u, dtype = 'int32')
        def get_item_features():
            descp,year,genre = [],[],[]
            for i in items:
                d, g, y = self.get_item_feature(i)
                descp.append(d)
                year.append(y)
                genre.append(g)
            return descp,year,genre
        descp,year,genre = get_item_features()
        
        descp = np.array(descp)
        year = np.array(year)
        genre = np.array(genre)
        m = ModelData()
    
        m.userids = users
        m.itemids = np.array(items)
        m.gtitem = gtItem
        m.descp = descp
        m.genre = genre
        m.year = year
        return m
    
    def gen_test_data(self):
        t1 = time()
            #test_data.append(m)
        
        if self.num_threads>1:
            pool = multiprocessing.Pool(processes=self.num_threads)
            res = pool.map(self._one_test_data, range(len(self.testData)))
            pool.close()
            pool.join()
            self.test_data = [r for r in res]
            
        else:    
            self.test_data = [self._one_test_data(_i) for _i in range(len(self.testData))]
        print("gen_test_data [%.1f s]"%(time()-t1))
        
            
    def gen_train_data(self):
        t1 = time()
        user_input, item_input, labels = [],[],[]
        item_des, item_year, item_genre = [],[],[]
        
        for index,row in self.trainData.iterrows():
            # positive instance
            u = row["UserID"]
            i = row["ItemID"]
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            d, g, y = self.get_item_feature(i)
            item_des.append(d)
            item_year.append(y)
            item_genre.append(g)
            # negative instances
            
            negatives = row["Negatives"]
            for _i in range(len(negatives)):
                neg_item_ID = negatives[_i]
                user_input.append(u)
                item_input.append(neg_item_ID)
                labels.append(0)
                d, g, y = self.get_item_feature(neg_item_ID)
                item_des.append(d)
                item_year.append(y)
                item_genre.append(g)
                
        user_input, item_input, labels, item_des, item_year, item_genre = shuffle(user_input, item_input, labels, item_des, item_year, item_genre)
        user_input = np.array(user_input)
        item_input = np.array(item_input)
        item_des = np.array(item_des)
        item_year = np.array(item_year)
        item_genre = np.array(item_genre)
        labels = np.array(labels)
        
        m = ModelData()
        
        m.userids = user_input
        m.itemids = item_input
        m.labels = labels
        m.descp = item_des
        m.genre = item_genre
        m.year = item_year
        
        self.train_data = m
        print("gen_train_data [%.1f s]"%(time()-t1))
    def save(self, path):
        self.trainData.to_pickle(path+".train.data")
        self.testData.to_pickle(path+".test.data")
        self.validData.to_pickle(path+".valid.data")
        
        self.pickleit(self.train_data, path+".train_data")
        self.pickleit(self.test_data, path+".test_data")
        
    def pickleit(self, o, path):
        
        with open(path, 'wb') as f:
            pickle.dump(o, f)
            
    def loadPickle(self, path):
        with open(path, 'rb') as f:
            o = pickle.load(f)
        return o
        