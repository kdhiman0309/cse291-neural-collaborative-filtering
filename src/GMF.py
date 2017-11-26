'''
Created on Aug 9, 2016

Keras Implementation of Generalized Matrix Factorization (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from Dataset import Dataset
from evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse
from sklearn.utils import shuffle


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def init_normal():
    return initializers.RandomNormal(stddev=0.01)


def get_model(num_users, num_items, latent_dim, regs=[0,0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[1]), input_length=1)   
    
 
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    
    # Element-wise product of user and item embeddings 
    predict_vector = keras.layers.multiply([user_latent, item_latent])
    
    dense1 = Dense(latent_dim, activation='relu', kernel_initializer='lecun_uniform', name = 'dense1')(predict_vector)
    dense2 = Dense(int(latent_dim/2), activation='relu', kernel_initializer='lecun_uniform', name = 'dense2')(dense1)
    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(dense2)
    
    model = Model(inputs=[user_input, item_input], 
                outputs=prediction)

    return model


def get_train_instances_original(dataset, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_items = dataset.num_items
    train_pairs = set(list(zip(dataset.trainData["UserID"].values, dataset.trainData["ItemID"].values)))
    for index,row in dataset.trainData.iterrows():
        # positive instance
        u = row["UserID"]
        i = row["ItemID"]
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train_pairs:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels
def get_train_instances(dataset, num_negatives):
    user_input, item_input, labels = [],[],[]
    for index,row in dataset.trainData.iterrows():
        # positive instance
        u = row["UserID"]
        i = row["ItemID"]
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        
        negatives = row["Negatives"]
        for _i in range(num_negatives):
            neg_item_ID = negatives[_i]
            user_input.append(u)
            item_input.append(neg_item_ID)
            labels.append(0)
    user_input, item_input, labels = shuffle(user_input, item_input, labels)
    
    return user_input, item_input, labels

def train(
    num_factors = 8,
    regs = [0,0],
    num_negatives = 4,
    learner = "adam",
    learning_rate = 0.001,
    epochs = 10,
    batch_size = 256,
    verbose = 1,
    out=0,
    topK = 10,
    datapath = "../data/movielens",
    prep_data=False
    ):
    evaluation_threads = 1 #mp.cpu_count()
    #print("GMF arguments: %s" %(args))
    model_out_file = 'Pretrain/%s_GMF_%d_%d.h5' %(datapath, num_factors, time())
    
    # Loading data
    t1 = time()
    
    dataset = Dataset(datapath, prep_data=prep_data)
    num_users, num_items = dataset.num_users, dataset.num_items
    
    # Build model
    model = get_model(num_users, num_items, num_factors, regs)
    
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        #model.compile(optimizer=Adam(), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    
    #model.compile(optimizer="adam", loss='binary_crossentropy')
    print(model.summary())
    
    # Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, dataset.trainData, dataset, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    #mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights())+np.linalg.norm(model.get_layer('item_embedding').get_weights())
    #p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
    print('Init Test: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))
       
    # Train model
    # Generate training instances
    _t = dataset.train_data
    user_input, item_input, labels = _t.userids, _t.itemids, _t.labels
    
    
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        
        # Training
        hist = model.fit([user_input, item_input], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, dataset.trainData, dataset, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if out > 0:
        print("The best GMF model is saved to %s" %(model_out_file))
    
if True:
    train(
        num_factors = 8,
        regs = [0,0],
        num_negatives = 5,
        learner = "adam",
        learning_rate = 0.001,
        epochs = 15,
        batch_size = 256,
        verbose = 1,
        out=0,
        topK = 10,
        datapath = "../data/movielens20M",
    		prep_data=True
        )
