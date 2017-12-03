'''
Created on Aug 9, 2016

Keras Implementation of Generalized Matrix Factorization (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import keras
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


def get_model(num_users, num_items, latent_dim, gensim_dim, regs=[0,0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    text_input_sparse = Input(shape=(gensim_dim,), name = 'text_input_sparse')
    
    item_feature_input_genre = Input(shape=(24,), name = 'item_feature_input_genre')
    item_feature_input_year = Input(shape=(6,), name = 'item_feature_input_year')
    genre_dense = Dense(8, activation='relu', kernel_initializer='lecun_uniform',
                            name = "genre_dense")(item_feature_input_genre)
    year_dense = Dense(4, activation='relu', kernel_initializer='lecun_uniform',
                            name = "year_dense")(item_feature_input_year)
    
    complete_item_features = keras.layers.concatenate([text_input_sparse, genre_dense,year_dense])

    dense_1 = Dense(int(latent_dim*2), activation='relu', kernel_initializer='lecun_uniform', name = 'dense_1')(complete_item_features)
    item_features_latent = Dense(latent_dim, activation='relu', kernel_initializer='lecun_uniform', name = 'dense2')(dense_1)
    
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[1]), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    
    
    multiply_layer0 = keras.layers.multiply([user_latent, item_latent])
    multiply_layer1 = keras.layers.multiply([user_latent, item_features_latent])
    
    multiply_layer = keras.layers.add([multiply_layer0,multiply_layer1])
    # Element-wise product of user and item embeddings 
    #item_user_layer = keras.layers.concatenate([user_latent,complete_item_features])
    
    
    #dense1 = Dense(latent_dim, activation='relu', kernel_initializer='lecun_uniform', name = 'dense1')(multiply_layer)
    #concat_two = keras.layers.concatenate([multiply_layer,dense1])
    
    #dense2 = Dense(int(latent_dim/2), activation='relu', kernel_initializer='lecun_uniform', name = 'dense2')(dense1)
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(multiply_layer)
    
    model = Model(inputs=[user_input, item_input, text_input_sparse, item_feature_input_genre, item_feature_input_year], 
                outputs=prediction)

    return model


class MyModel():
    def train(self,
        gensim_dim = 10,
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
        model_out_file = '../../model/GMF_%d_%d.h5'%(num_factors, time())
        
        # Loading data
        t1 = time()
        
        dataset = Dataset(datapath, prep_data=prep_data,gensim_dim=gensim_dim)
        num_users, num_items = dataset.num_users, dataset.num_items
        
        # Build model
        model = get_model(num_users, num_items, num_factors, gensim_dim, regs)
        
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
        #print(model.summary())
        
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
        _t2 = dataset.train_data_item_feat
        hist = model.fit([_t.userids, _t.itemids, _t2.descp, _t2.genre, _t2.year], _t.labels, batch_size=batch_size,
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
        
        num_factors = 8,
        gensim_dim = 10,
        regs = [0.000001,0.000001],
        num_negatives = 5,
        learner = "adam",
        learning_rate = 0.001,
        epochs = 10,
        batch_size = 256,
        verbose = 1,
        out=1,
        topK = 10,
        datapath = "../../data/movielens20M",
    		prep_data=True
        )
