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

def get_gmf_pairwise_base_model(latent_dim):
    input1 = Input(shape=(latent_dim,), dtype='int32', name = 'input1')
    input2 = Input(shape=(latent_dim,), dtype='int32', name = 'input2')
    
    # Element-wise product of user and item embeddings 
    predict_vector = keras.layers.multiply([input1, input2])
    
    # Final prediction layer
    prediction = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(predict_vector)
    
    model = Model(inputs=[input1, input2], 
                outputs=prediction)
    return model

def get_gmf_pairwise_base_model2(latent_dim):
    input1 = Input(shape=(latent_dim,), dtype='int32', name = 'input1')
    input2 = Input(shape=(latent_dim,), dtype='int32', name = 'input2')
    
    # Element-wise product of user and item embeddings 
    predict_vector = keras.layers.multiply([input1, input2])
    
    
    model = Model(inputs=[input1, input2], 
                outputs=predict_vector)
    return model

def get_gmf_pairwise_model(num_users,num_items,latent_dim,regs=[0,0]):
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item1_input = Input(shape=(1,), dtype='int32', name = 'item1_input')
    item2_input = Input(shape=(1,), dtype='int32', name = 'item2_input')

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[1]), input_length=1)   
    user_latent = Flatten(name="user_latent")(MF_Embedding_User(user_input))
    item1_latent = Flatten(name="item1_latent")(MF_Embedding_Item(item1_input))
    item2_latent = Flatten(name="item2_latent")(MF_Embedding_Item(item2_input))
    model =  get_gmf_pairwise_base_model(latent_dim)
    prediction1 = model([user_latent,item1_latent])
    prediction2 = model([user_latent,item2_latent])
    subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],
                        output_shape=lambda shapes: shapes[0])([prediction1,prediction2])
    #prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(keras.layers.concatenate([prediction1,prediction2]))
    prediction = Activation('sigmoid')(subtract_layer)
    pairwise_model = Model(inputs=[user_input, item1_input, item2_input],outputs=prediction)
    inference_model = Model(inputs=[user_input, item1_input, item2_input],outputs=prediction1)
    print(pairwise_model.summary())
    return pairwise_model,inference_model 

def get_gmf_pairwise_model2(num_users,num_items,latent_dim,regs=[0,0]):
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item1_input = Input(shape=(1,), dtype='int32', name = 'item1_input')
    item2_input = Input(shape=(1,), dtype='int32', name = 'item2_input')

    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[1]), input_length=1)   
    user_latent = Flatten(name="user_latent")(MF_Embedding_User(user_input))
    item1_latent = Flatten(name="item1_latent")(MF_Embedding_Item(item1_input))
    item2_latent = Flatten(name="item2_latent")(MF_Embedding_Item(item2_input))
    model =  get_gmf_pairwise_base_model2(latent_dim)
    prediction1 = model([user_latent,item1_latent])
    prediction2 = model([user_latent,item2_latent])
    #subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],
    #                    output_shape=lambda shapes: shapes[0])([prediction1,prediction2])
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(keras.layers.concatenate([prediction1,prediction2]))
    #prediction = Activation('sigmoid')(subtract_layer)
    pairwise_model = Model(inputs=[user_input, item1_input, item2_input],outputs=prediction)
    inference_model = Model(inputs=[user_input, item1_input, item2_input],outputs=prediction1)
    print(pairwise_model.summary())
    return pairwise_model,inference_model 
    
def get_model(num_users, num_items, latent_dim, regs=[0,0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item1_input = Input(shape=(1,), dtype='int32', name = 'item1_input')
    item2_input = Input(shape=(1,), dtype='int32', name = 'item2_input')


    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[1]), input_length=1)   
    
 
    # Crucial to flatten an embedding vector!
    user_latent = Flatten(name="user_latent")(MF_Embedding_User(user_input))
    item1_latent = Flatten(name="item1_latent")(MF_Embedding_Item(item1_input))
    item2_latent = Flatten(name="item2_latent")(MF_Embedding_Item(item2_input))
    
    # Element-wise product of user and item embeddings 
    predict_vector_1 = keras.layers.multiply([user_latent, item1_latent])
    predict_vector_2 = keras.layers.multiply([user_latent, item2_latent])

    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    dense_1 = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'dense_1')
    prediction_1 = dense_1(predict_vector_1)
    prediction_2 = dense_1(predict_vector_2)

    subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],
                        output_shape=lambda shapes: shapes[0])([prediction_1,prediction_2])
    prediction = Activation('sigmoid')(subtract_layer)
    
    model = Model(inputs=[user_input, item1_input, item2_input], 
                outputs=prediction)
    inference_model = Model(inputs=[user_input, item1_input, item2_input],outputs=prediction_1)

    print(model.summary())
    return model,inference_model


def get_model_sep(num_users, num_items, latent_dim, regs=[0,0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item1_input = Input(shape=(1,), dtype='int32', name = 'item1_input')
    item2_input = Input(shape=(1,), dtype='int32', name = 'item2_input')


    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[0]), input_length=1)
    MF_Embedding_Item_1 = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding_1',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[1]), input_length=1)   
    
    MF_Embedding_Item_2 = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding_2',
                                  embeddings_initializer = init_normal(), embeddings_regularizer= l2(regs[1]), input_length=1)   
 
    # Crucial to flatten an embedding vector!
    user_latent = Flatten(name="user_latent")(MF_Embedding_User(user_input))
    item1_latent = Flatten(name="item1_latent")(MF_Embedding_Item_1(item1_input))
    item2_latent = Flatten(name="item2_latent")(MF_Embedding_Item_2(item2_input))
    
    # Element-wise product of user and item embeddings 
    predict_vector_1 = keras.layers.multiply([user_latent, item1_latent])
    predict_vector_2 = keras.layers.multiply([user_latent, item2_latent])

    
    # Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    dense_1 = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'dense_1')
    dense_2 = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'dense_2')

    prediction_1 = dense_1(predict_vector_1)
    prediction_2 = dense_2(predict_vector_2)

    subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],
                        output_shape=lambda shapes: shapes[0])([prediction_1,prediction_2])
    prediction = Activation('sigmoid')(subtract_layer)
    
    model = Model(inputs=[user_input, item1_input, item2_input], 
                outputs=prediction)
    inference_model = Model(inputs=[user_input, item1_input, item2_input],outputs=prediction_1)

    print(model.summary())
    return model,inference_model

def get_train_instances_original(dataset, num_negatives):
    user_input, item_input, labels = [],[],[]
    item_input_2 = []
    num_items = dataset.num_items
    train_pairs = set(list(zip(dataset.trainData["UserID"].values, dataset.trainData["ItemID"].values)))
    for index,row in dataset.trainData.iterrows():
        # positive instance
        u = row["UserID"]
        i = row["ItemID"]
        
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train_pairs:
                j = np.random.randint(num_items)
            user_input.append(u)
            if t%2 == 0:
               item_input_2.append(j)     
               item_input.append(i)
               labels.append(1)
            else:
               item_input_2.append(i)     
               item_input.append(j)
               labels.append(0)
                
    return user_input, item_input,item_input_2, labels

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

class MyModel():
    def train(self,
        num_factors=8,
        data_path="../../data/movielens20M",
        num_epochs = 20,
        batch_size = 256,
        mf_dim = 8,
        layers = [32,16,8],
        reg_mf = 0,
        reg_layers = [0,0,0],
        num_negatives = 4,
        learning_rate = 0.001,
        learner = "adam",
        verbose = 1,
        mf_pretrain = '',
        mlp_pretrain = '',
        out=0,
        prep_data=False
        ):
        evaluation_threads = 1 #mp.cpu_count()
        model_out_file = '../../model/GMF_%d_%d.h5' %(num_factors, time())
        
        # Loading data
        t1 = time()
        
        dataset = Dataset(data_path,prep_data=prep_data)
        trainData, testData = dataset.train_data, dataset.testData
        num_users, num_items = dataset.num_users, dataset.num_items
        return
        # Build model
        #model,inference_model = get_gmf_pairwise_model()
        model,inference_model = get_gmf_pairwise_model(num_users, num_items, num_factors)
        #model,inference_model = get_model_sep(num_users, num_items, num_factors)

        if learner.lower() == "adagrad": 
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
            #model.compile(optimizer=Adam(), loss='binary_crossentropy')
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
        
        topK=10
        def evaulate(_data):
            (hits, ndcgs, aucs) = evaluate_model(model, _data, dataset, topK, evaluation_threads)
            return np.array(hits).mean(), np.array(ndcgs).mean(), np.array(aucs).mean()
        
        # Init performance
        t1 = time()
        hr, ndcg, auc = evaulate(dataset.testData)
        print('Init Test: HR = %.4f, NDCG = %.4f, AUC = %.4f\t [%.1f s]' % (hr, ndcg, auc, time()-t1))
        
        _t = dataset.train_data
        user_input, item1_input,item2_input, labels = _t.userids, _t.item1ids,_t.item2ids, _t.labels        
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
        hist = model.fit([user_input, item1_input, item2_input], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=num_epochs, verbose=2, shuffle=True, callbacks=[metricsClbk])
        
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
        
        if out > 0:
            print("The best NeuMF model is saved to %s" %(model_out_file))
            
        self.dataset = dataset
        self.best_model = model

if True:
    m = MyModel()
    m.train(
        num_factors=8,
        data_path="../../data/movielens20M",
        num_epochs = 10,
        batch_size = 256,
        mf_dim = 8,
        layers = [32,16,8],
        reg_mf = 0,
        reg_layers = [0,0,0],
        num_negatives = 4,
        learning_rate = 0.001,
        learner = "adam",
        verbose = 1,
        mf_pretrain = '',
        mlp_pretrain = '',
        out=0,
        prep_data=True
    )

