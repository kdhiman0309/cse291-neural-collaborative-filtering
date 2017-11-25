'''
Created on Aug 8, 2016
Processing datasets. 


'''
import scipy.sparse as sp
import numpy as np
import pandas as pd
from collections import defaultdict
import copy
class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path, prep_data=False,count_per_user_test=1,count_per_user_validation=0,num_negatives_train=4):
        '''
        Constructor
        '''
        self.item_features = self.load_file(path + ".itemfeatures.pkl")
        if prep_data:
            self.item_list = list(self.item_features.index)
            self.trainData = self.load_file(path+".ratings.data.pkl")
            self.split_train_test(count_per_user_test=count_per_user_test,count_per_user_validation=count_per_user_validation)
            self.negative_sampling(num_negatives_train=num_negatives_train)
        else:
            self.trainData = self.load_file(path + ".train.data")
            self.testData = self.load_file(path + ".test.data")
            #self.validData = self.load_file(path + ".valid.data")
            self.validData = self.testData
        
        
        self.num_users = self.trainData["UserID"].max()+1
        self.num_items = self.trainData["ItemID"].max()+1
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
        
    def save(self, path):
        self.trainData.to_pickle(path+".train.data")
        self.testData.to_pickle(path+".test.data")
        self.validData.to_pickle(path+".valid.data")

    def get_item_feature(self,item_input):
       text_features = []
       other_features = []
       for i in item_input:
          curr_row = self.item_features.loc[i]
          text_features.append(curr_row['description_encoded'])
          other_features.append(curr_row['year_encoded']+curr_row["genre_encoded"])
       return text_features,other_features
