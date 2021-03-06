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
    print(user_input)
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
    item_des, item_year, item_genre = [],[],[]
    
    for index,row in dataset.trainData.iterrows():
        # positive instance
        u = row["UserID"]
        i = row["ItemID"]
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        d, g, y = dataset.get_item_feature(i)
        item_des.append(d)
        item_year.append(y)
        item_genre.append(g)
        # negative instances
        
        negatives = row["Negatives"]
        for _i in range(num_negatives):
            neg_item_ID = negatives[_i]
            user_input.append(u)
            item_input.append(neg_item_ID)
            labels.append(0)
            d, g, y = dataset.get_item_feature(neg_item_ID)
            item_des.append(d)
            item_year.append(y)
            item_genre.append(g)
            
    user_input, item_input, labels, item_des, item_year, item_genre = shuffle(user_input, item_input, labels, item_des, item_year, item_genre)
    return user_input, item_input, labels, item_des, item_year, item_genre

def train(
    data_path="data",
    num_epochs = 20,
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
    evaluation_threads = 1#mp.cpu_count()
    print("NeuMF arguments:")
    model_out_file = '../pretrain/%s_NeuMF_%d_%s_%d.h5' %(data_path, mf_dim, layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(data_path,prep_data=prep_data)
    trainData, validData, testData = dataset.trainData, dataset.validData, dataset.testData
    num_users, num_items = dataset.num_users, dataset.num_items
    
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, len(trainData), len(testData)))
    
    # Build model
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    print(model.summary())
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
    (hits, ndcgs) = evaluate_model(model, dataset, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if out > 0:
        model.save_weights(model_out_file, overwrite=True) 
    
    # Generate training instances
    t1 = time()
     
    #user_input, item_input, labels, item_des, item_year, item_genre = get_train_instances(dataset, num_negatives)
    #print("training instances [%.1f s]"%(time()-t1))
    _t = dataset.train_data
    user_input, item_input, labels, item_des, item_year, item_genre = \
    _t.userids, _t.itemids, _t.labels, _t.descp, _t.year, _t.genre
    
    # Training model
    for epoch in range(num_epochs):
        t1 = time()
        
        # Training
        hist = model.fit([user_input, item_input], #input
                         labels, # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, dataset, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if out > 0:
        print("The best NeuMF model is saved to %s" %(model_out_file))


train(
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
    prep_data=True
)
