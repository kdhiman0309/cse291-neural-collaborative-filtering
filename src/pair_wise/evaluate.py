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

def evaluate_model(model, testData, K, num_thread):
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

    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating_pairwise_mp, range(len(testData)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for index, row in testData.iterrows():
        (hr,ndcg) = eval_one_rating_pairwise(row, model, K)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

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
     top_k.append(item)
     i = len(top_k)-1
          
     while i>0:
          #score = get_pairwise_score(model,user,top_k[i],top_k[i-1])
          score = scores[i-1]
          if score < 0.5:
               break
          else:
               tmp = top_k[i]
               top_k[i] = top_k[i-1]
               top_k[i-1] = tmp
          i=i-1
     return top_k
     
def eval_one_rating_pairwise(row, model, K):
    items = row["Negatives"]#_testNegatives[idx]
    u = row["UserID"]
    gtItem = row["ItemID"]
    items.append(gtItem)
    shuffle(items)
  
    # Get prediction scores
    top_k = []
    top_k.append(items[0])

    for i in range(1,len(items)):    
         top_k = sift_up(model,u,top_k,items[i])
         if(len(top_k) > K):
              top_k = top_k[0:K]

    # Evaluate top rank list
    #print(len(top_k))
    #print(top_k)
    #print(gtItem)
    hr = getHitRatio(top_k, gtItem)
    ndcg = getNDCG(top_k, gtItem)
    return (hr, ndcg)
