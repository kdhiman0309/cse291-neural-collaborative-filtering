'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''

import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
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

def get_model(num_users, num_items, layers = [20,10], reg_layers=[0,0],item_feature_merge=-1):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    text_input_sparse = Input(shape=(30,), name = 'text_input_sparse')
    
    item_feature_input_genre = Input(shape=(24,), name = 'item_feature_input_genre')
    item_feature_input_year = Input(shape=(6,), name = 'item_feature_input_year')
    genre_dense = Dense(8, activation='relu', kernel_initializer='lecun_uniform',
                            name = "genre_dense")(item_feature_input_genre)
    year_dense = Dense(4, activation='relu', kernel_initializer='lecun_uniform',
                            name = "year_dense")(item_feature_input_year)
    
    complete_item_features = keras.layers.concatenate([text_input_sparse, genre_dense,year_dense])
    latent_dim = int(layers[0]/2)
    dense_1 = Dense(int(latent_dim*2), activation='relu', kernel_initializer='lecun_uniform', name = 'dense_1')(complete_item_features)
    item_features_latent = Dense(latent_dim, activation='relu', kernel_initializer='lecun_uniform', name = 'dense2')(dense_1)
    complete_item_features = keras.layers.concatenate([tfidf_dense, genre_dense,year_dense])

    
    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = 'user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    
    # The 0-th layer is the concatenation of embedding layers
    vector = keras.layers.concatenate([user_latent, item_latent, item_features_latent])
    
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        if idx == item_feature_merge:
            vector = layer(keras.layers.concatenate([vector,complete_item_features]))
        else:
            vector = layer(vector)
        
    # Final prediction layer
    if item_feature_merge == -1:
        vector = keras.layers.concatenate([vector,complete_item_features])

    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    
    model = Model(inputs=[user_input, item_input, text_input_sparse, item_feature_input_genre, item_feature_input_year], 
                  outputs=prediction)
    
    return model

class MyModel():
    def train(self,

    def train(self,
        num_factors = 8,
        layers = [32,16,8],
        item_feature_merge = -1,
        reg_layers = [0,0,0,0],
        num_negatives = 4,
        learner = "adam",
        learning_rate = 0.001,
        epochs = 10,
        batch_size = 256,
        verbose = 1,
        out=0,
        topK = 10,
        datapath = "../data/movielens"
        prep_data = False
        ):
        
        topK = 10
        evaluation_threads = 1 #mp.cpu_count()
        #print("MLP arguments: %s " %(args))
        model_out_file = '../../model/MLP_%d_%d.h5'%(num_factors, time())
            
        # Loading data
        t1 = time()
        dataset = Dataset(datapath)
        #trainData, validData, testData = dataset.trainData, dataset.validData, dataset.testData
        num_users, num_items = dataset.num_users, dataset.num_items
        #print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
        #      %(time()-t1, num_users, num_items, len(trainData), len(testData)))
        
        # Build model
        model = get_model(num_users, num_items, layers, reg_layers)
        if learner.lower() == "adagrad": 
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    
        
        # Init performance
        t1 = time()
        
        def evaulate(_data):
            (hits, ndcgs, aucs) = evaluate_model(model, _data, dataset, topK, evaluation_threads)
            return np.array(hits).mean(), np.array(ndcgs).mean(), np.array(aucs).mean()
        hr, ndcg, auc = evaulate(dataset.testData)
        print('Init Test: HR = %.4f, NDCG = %.4f, AUC = %.4f\t [%.1f s]' % (hr, ndcg, auc, time()-t1))
        #hr, ndcg, auc = evaulate(dataset.testColdStart)
        #print('Cold Start: HR = %.4f, NDCG = %.4f, AUC = %.4f\t [%.1f s]' % (hr, ndcg, auc, time()-t1))
        # Train model
        # Generate training instances
        _t = dataset.train_data
        user_input, item_input, labels = _t.userids, _t.itemids, _t.labels
        
        best_hr, best_ndcg, best_iter,epoch = hr, ndcg, -1,0
        
        class MetricsCallback(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.epoch = 0
                self.best_hr = 0
                self.best_ndcg = 0
                self.best_iter = -1
            def on_epoch_end(self, batch, logs={}):
                
        class MetricsCallback(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.epoch = 0
                self.best_hr = 0
                self.best_ndcg = 0
                self.best_iter = -1
            def on_epoch_end(self, batch, logs={}):
                
                t2 = time()
                hr, ndcg, auc = evaulate(dataset.testData)
                print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, AUC = %.4f [%.1f s]' 
                      % (self.epoch,  t2-t1, hr, ndcg, auc, time()-t2))
                if hr > self.best_hr:
                    self.best_hr, self.best_ndcg, self.best_iter = hr, ndcg, self.epoch
                    model.save(model_out_file, overwrite=True)
                self.epoch+=1
        
        t1 = time()
        metricsClbk = MetricsCallback()
                t2 = time()
                hr, ndcg, auc = evaulate(dataset.testData)
                print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, AUC = %.4f [%.1f s]' 
                      % (self.epoch,  t2-t1, hr, ndcg, auc, time()-t2))
                if hr > self.best_hr:
                    self.best_hr, self.best_ndcg, self.best_iter = hr, ndcg, self.epoch
                    model.save(model_out_file, overwrite=True)
                self.epoch+=1
        
        t1 = time()
        metricsClbk = MetricsCallback()
        # Training
        #hist = model.fit_generator(dataset.generator_train_data(batch_size),steps_per_epoch=1+int((len(dataset.train_data.userids)/batch_size)),
        #                          epochs=epochs, verbose=2, shuffle=True, callbacks=[metricsClbk])
        _t = dataset.train_data
        hist = model.fit([_t.userids, _t.itemids, _t.descp, _t.genre, _t.year], _t.labels, batch_size=batch_size,
                                  epochs=epochs, verbose=2, shuffle=True, callbacks=[metricsClbk])
		# E
        #Evaluation                
        self.model = model
        model = keras.models.load_model(model_out_file)
        t2 = time()
        hr, ndcg, auc = evaulate(dataset.testData)
        print('Test [%.1f s]: HR = %.4f, NDCG = %.4f, AUC = %.4f, [%.1f s]' 
              % (t2-t1, hr, ndcg, auc, time()-t2))
        t2 = time()
        hr, ndcg, auc = evaulate(dataset.testColdStart)
        print('Cold Start [%.1f s]: HR = %.4f, NDCG = %.4f, AUC = %.4f, [%.1f s]' 
              % (t2-t1, hr, ndcg, auc, time()-t2))
        t2 = time()
        hr, ndcg, auc = evaulate(dataset.testColdStartPseudo)
        print('Cold Start Pseudo [%.1f s]: HR = %.4f, NDCG = %.4f, AUC = %.4f, [%.1f s]' 
              % (t2-t1, hr, ndcg, auc, time()-t2))
        
        if out > 0:
            print("The best GMF model is saved to %s" %(model_out_file))
            
        self.dataset = dataset
        self.best_model = model
        
if True:
    m = MyModel()
    m.train(

if True:
    m = MyModel()
    m.train(
        num_factors = 8,
        layers = [64,32,16,8],
    item_feature_merge = -1,
        reg_layers = [0,0,0,0],
        num_negatives = 4,
        learner = "adam",
        learning_rate = 0.001,
        epochs = 10,
        batch_size = 256,
        verbose = 1,
        out=0,
        topK = 10,
        datapath = "../data/movielens"
    prep_data = False
        )
