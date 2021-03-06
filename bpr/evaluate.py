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
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
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

    hits, ndcgs, aucs = [],[],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating_mp, testData.index)
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        aucs = [r[2] for r in res]
        return (hits, ndcgs, aucs)
    # Single thread
    for index,row in testData.iterrows():
        (hr,ndcg, auc) = eval_one_rating(row, model, K)
        hits.append(hr)
        ndcgs.append(ndcg)    
        aucs.append(auc)
    return (hits, ndcgs, aucs)

def eval_one_rating_mp(idx):
     return eval_one_rating(_testData.loc[idx],_model,_K)

def eval_one_rating(row, model, K):
    
    items = row["Negatives"]#_testNegatives[idx]
    u = row["UserID"]
    gtItem = row["ItemID"]
    items.append(gtItem)
    # Get prediction scores
    predictions = [0]*len(items)
    for i in range(len(items)):
        predictions[i] = model.predict(u, items[i])
        
    map_item_score = {}
    user_auc = 0.0
    gtitem_score = predictions[-1]
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
        user_auc += 1.0 if predictions[i] < gtitem_score else 0.0
    items.pop()
    user_auc = user_auc/len(items)
    # Evaluate top rank list
    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg, user_auc)

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
