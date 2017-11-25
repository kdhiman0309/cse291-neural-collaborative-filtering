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
    datapath = "../data/movielens"
    ):
    evaluation_threads = 6 #mp.cpu_count()
    #print("GMF arguments: %s" %(args))
    model_out_file = 'Pretrain/%s_GMF_%d_%d.h5' %(datapath, num_factors, time())
    
    # Loading data
    t1 = time()
    
    dataset = Dataset(datapath)
    trainData, validData, testData = dataset.trainData, dataset.validData, dataset.testData
    num_users, num_items = dataset.num_users, dataset.num_items
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, len(trainData), len(testData)))
    
    # Build model
    model,inference_model = get_model(num_users, num_items, num_factors, regs)
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
    (hits, ndcgs) = evaluate_model(model, testData, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    #mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights())+np.linalg.norm(model.get_layer('item_embedding').get_weights())
    #p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
    print('Init Test: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))
    
    #(hits, ndcgs) = evaluate_model(model, validData, topK, evaluation_threads)
    #hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    #print('Init Valid: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))
    
    # Train model
    # Generate training instances
    user_input, item_input,item_input_2, labels = get_train_instances_original(dataset, num_negatives)
    user_input = np.array(user_input)
    item_input = np.array(item_input)
    item_input_2 = np.array(item_input_2)
    labels = np.array(labels)
    
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        
        # Training
        hist = model.fit([user_input, item_input, item_input_2], #input
                         labels, # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %verbose == 0:
            #(hits, ndcgs) = evaluate_model(model, validData, topK, evaluation_threads)
            (hits_test, ndcgs_test) = evaluate_model(model, testData, topK, evaluation_threads)
    
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            hr_test, ndcg_test = np.array(hits_test).mean(), np.array(ndcgs_test).mean()
            print('Iteration %d [%.1f s]: (Valid) HR = %.4f, NDCG = %.4f, (Test) HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, 0, 0, hr_test, ndcg_test, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if out > 0:
        print("The best GMF model is saved to %s" %(model_out_file))
    

train(
    num_factors = 8,
    regs = [0,0],
    num_negatives = 8,
    learner = "adam",
    learning_rate = 0.001,
    epochs = 15,
    batch_size = 256,
    verbose = 1,
    out=0,
    topK = 10,
    datapath = "../../data/movielens"
    )

