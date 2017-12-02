'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Reshape, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import GMF, MLP
import argparse
from sklearn.utils import shuffle
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
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
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()

def init_normal():
    return initializers.RandomNormal(stddev=0.01)

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    shared_embedding = False
    # Embedding layer
    #MF_Embedding_User = Embedding(num_users, 8, input_length=1)
    if not shared_embedding:
    
        MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                      embeddings_initializer = init_normal(), embeddings_regularizer= l2(reg_mf), input_length=1)
        MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                      embeddings_initializer = init_normal(), embeddings_regularizer= l2(reg_mf), input_length=1)   
        MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = "mlp_embedding_user",
                                      embeddings_initializer = init_normal(), embeddings_regularizer= l2(reg_layers[0]), input_length=1)
        MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'mlp_embedding_item',
                                      embeddings_initializer = init_normal(), embeddings_regularizer= l2(reg_layers[0]), input_length=1)   
        
    else:
        MF_Embedding_User = Embedding(input_dim = num_users, output_dim = max(mf_dim,int(layers[0]/2)), name = 'mf_embedding_user',
                                      embeddings_initializer = init_normal(), embeddings_regularizer= l2(reg_mf), input_length=1)
        MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = max(mf_dim,int(layers[0]/2)), name = 'mf_embedding_item',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(reg_mf), input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = keras.layers.multiply([mf_user_latent, mf_item_latent]) # element-wise multiply

    if not shared_embedding:
        # MLP part 
        mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
        mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
        mlp_vector = keras.layers.concatenate([mlp_user_latent, mlp_item_latent])
    else:
        mlp_vector = keras.layers.concatenate([mf_user_latent, mf_item_latent])
    
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = keras.layers.concatenate([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(inputs=[user_input, item_input], 
                  outputs=prediction)
    print(model.summary())
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
    
    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
    return model

class MyModel():
    def train(self,
        datapath="data",
        epochs = 20,
        batch_size = 256,
        mf_dim = 8,
        layers = [64,32,16,8],
        reg_mf = 0,
        reg_layers = [0,0,0,0],
        num_negatives = 5,
        learning_rate = 0.001,
        learner = "adam",
        verbose = 1,
        mf_pretrain = '',
        mlp_pretrain = '',
        out=0,
        prep_data=False
        ):
        
        topK = 10
        evaluation_threads = 1 #mp.cpu_count()
        #print("MLP arguments: %s " %(args))
        model_out_file = '../model/NeuMF_%d_%d_%d.h5' %(mf_dim, layers[0], time())
        
        # Loading data
        t1 = time()
        
        dataset = Dataset(datapath, prep_data=prep_data)
        num_users, num_items = dataset.num_users, dataset.num_items
        
        # Build model
        model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
        #print(model.summary())
        if learner.lower() == "adagrad": 
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
        
        # Load pretrain model
        if mf_pretrain != '' and mlp_pretrain != '':
            gmf_model = GMF.get_model(num_users,num_items,mf_dim)
            gmf_model.load_weights(mf_pretrain)
            mlp_model = MLP.get_model(num_users,num_items, layers, reg_layers)
            mlp_model.load_weights(mlp_pretrain)
            model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
            print("Load pretrained GMF (%s) and MLP (%s) models done. " %(mf_pretrain, mlp_pretrain))
            
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
        hist = model.fit([user_input, item_input], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=epochs, verbose=2, shuffle=True, callbacks=[metricsClbk])
        
            # Evaluation
                
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
        
        print("The best NeuMF model is saved to %s" %(model_out_file))
            
        self.dataset = dataset
        self.best_model = model
       

if True:
    m = MyModel()
    m.train(
        epochs = 10,
        batch_size = 256,
        mf_dim = 16,
        layers = [32,16,8],
        reg_mf = 0.000001,
        reg_layers = [0.000001,0,0],
        num_negatives = 4,
        learning_rate = 0.001,
        learner = "adam",
        verbose = 1,
        mf_pretrain = '',
        mlp_pretrain = '',
        datapath="../data/movielens20M",
        prep_data=False
    )
