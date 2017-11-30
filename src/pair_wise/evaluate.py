'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
from random import shuffle
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testData = None
_K = None

def evaluate_model(model, testData, dataset, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testData
    global _K
    _model = model
    _testData = testData
    _K = K

    hits, ndcgs, aucs = [],[],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating_pairwise_mp, testData.index)
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        aucs = [r[2] for r in res]
        return (hits, ndcgs,aucs)
    # Single thread
    for index, row in testData.iterrows():
        (hr,ndcg,auc) = eval_one_rating_pairwise(row, model, K)
        hits.append(hr)
        ndcgs.append(ndcg)
        aucs.append(auc)
    return (hits, ndcgs,aucs)

def eval_one_rating_pairwise_mp(idx):
     return eval_one_rating_pairwise(_testData.loc[idx],_model,_K)
     
     
     
def eval_one_rating(row, model, K):
    items = row["Negatives"]#_testNegatives[idx]
    u = row["UserID"]
    gtItem = row["ItemID"]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = model.predict([users, np.array(items),np.array(items)], 
                                 batch_size=128, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0


def get_pairwise_score(model,user,item1,item2):
     return model.predict([np.array([user]), np.array([item1]),np.array([item2])], 
                                 batch_size=1, verbose=0)

def get_pairwise_score_batch(model,user,item1,item2):
     return model.predict([np.array(user), np.array(item1),np.array(item2)], 
                                 batch_size=min(128,len(user)), verbose=0)

     
def sift_up(model,user,top_k,item):
     if len(top_k)==0:     
          top_k.append(item)
          return top_k
          
     scores = get_pairwise_score_batch(model,[user]*len(top_k),[item]*len(top_k),top_k)
     i = 0
     while i<len(top_k):
        score = scores[i]
        if score < 0.5:
           i=i+1
           continue
        else:
           top_k.insert(i,item)
           break
        i=i+1
     return top_k
     
def eval_one_rating_pairwise(row, model, K):
    hr,ndcg,auc = eval_one_rating_pairwise_auc(row,model,K)
    return (hr, ndcg,auc)


def eval_one_rating_pairwise_auc(row, model, K):
    items = row["Negatives"]#_testNegatives[idx]
    u = row["UserID"]
    gtItem = row["ItemID"]
    predictions = model.predict([np.full(len(items),u,dtype=int), np.full(len(items),gtItem,dtype=int),np.array(items)], 
                                 batch_size=min(128,len(items)), verbose=0)
    
    #predictions = model.predict([np.full(len(items),u,dtype=int), np.array(items), np.full(len(items),gtItem,dtype=int)], 
     #                            batch_size=min(128,len(items)), verbose=0)

    rank = len(items) - np.sum(predictions > 0.5)
    hr = 0
    ndcg=0
    if rank < K:
        hr=1
        ndcg = math.log(2) / math.log(rank+2)
    return hr,ndcg,np.sum(predictions > 0.5)/len(items)
