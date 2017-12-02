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
from evaluate import evaluate_model, evaluate_model_inference
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
    prediction_inference = Activation('sigmoid')(prediction1)
    inference_model = Model(inputs=[user_input, item1_input],
                            outputs=prediction_inference)
    print(pairwise_model.summary())
    return pairwise_model,inference_model 

def get_mlp_base_model(latent_dim,num_layers,layers = [20,10], reg_layers=[0,0]):
    input1 = Input(shape=(latent_dim,), dtype='int32', name = 'input1')
    input2 = Input(shape=(latent_dim,), dtype='int32', name = 'input2')
    vector = keras.layers.concatenate([input1, input2])
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    model = Model(inputs=[input1, input2], 
                  outputs=prediction)
    return model
    

def get_mlp_model(num_users, num_items, layers = [20,10], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layers = len(layers) #Number of layers in the MLP

    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item1_input = Input(shape=(1,), dtype='int32', name = 'item1_input')
    item2_input = Input(shape=(1,), dtype='int32', name = 'item2_input')

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = 'user_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'item_embedding',
                                  embeddings_initializer = init_normal(), embeddings_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item1_latent = Flatten()(MLP_Embedding_Item(item1_input))
    item2_latent = Flatten()(MLP_Embedding_Item(item2_input))

    model =  get_mlp_base_model(latent_dim,num_layers,layers,reg_layers)
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

def get_neumf_base_model(num_users, num_items, mf_dim, layers = [20,10], reg_layers=[0,0]):
    num_layers = len(layers)
    gmf_input1 = Input(shape=(mf_dim,), dtype='int32', name = 'input1')
    gmf_input2 = Input(shape=(mf_dim,), dtype='int32', name = 'input2')

    mlp_input1 = Input(shape=(int(layers[0]/2),), dtype='int32', name = 'input1')
    mlp_input2 = Input(shape=(int(layers[0]/2),), dtype='int32', name = 'input2')

    mf_vector = keras.layers.multiply([gmf_input1, gmf_input2])
    mlp_vector = keras.layers.concatenate([mlp_input1, mlp_input2])
    
    for idx in range(1, num_layers):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    predict_vector = keras.layers.concatenate([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, activation='linear', kernel_initializer='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(inputs=[gmf_input1, gmf_input2, mlp_input1, mlp_input2], 
                  outputs=prediction)
    return model

    
    
def get_neumf_model(num_users, num_items, mf_dim, layers = [20,10], reg_layers=[0,0], mf_reg=0):
    assert len(layers) == len(reg_layers)
    num_layers = len(layers) #Number of layers in the MLP

    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item1_input = Input(shape=(1,), dtype='int32', name = 'item1_input')
    item2_input = Input(shape=(1,), dtype='int32', name = 'item2_input')

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



    # Crucial to flatten an embedding vector!
    user_mf_latent = Flatten()(MF_Embedding_User(user_input))
    item1_mf_latent = Flatten()(MF_Embedding_Item(item1_input))
    item2_mf_latent = Flatten()(MF_Embedding_Item(item2_input))

    if not shared_embedding:
        user_mlp_latent = Flatten()(MLP_Embedding_User(user_input))
        item1_mlp_latent = Flatten()(MLP_Embedding_Item(item1_input))
        item2_mlp_latent = Flatten()(MLP_Embedding_Item(item2_input))
    else:
        user_mlp_latent = user_mf_latent
        item1_mlp_latent = item1_mf_latent
        item2_mlp_latent = item2_mf_latent

    neumf_base_model =  get_neumf_base_model(num_users, num_items, mf_dim,layers,reg_layers)
    prediction1 = neumf_base_model([user_mf_latent,item1_mf_latent,user_mlp_latent,item1_mlp_latent])
    prediction2 = neumf_base_model([user_mf_latent,item1_mf_latent,user_mlp_latent,item1_mlp_latent])
    subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],
                        output_shape=lambda shapes: shapes[0])([prediction1,prediction2])

    #prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(keras.layers.concatenate([prediction1,prediction2]))
    prediction = Activation('sigmoid')(subtract_layer)
    pairwise_model = Model(inputs=[user_input, item1_input, item2_input],outputs=prediction)
    inference_model = Model(inputs=[user_input, item1_input, item2_input],outputs=prediction1)
    print(pairwise_model.summary())
    return pairwise_model,inference_model
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
    prediction_inference = Activation('sigmoid')(prediction_1)
    model = Model(inputs=[user_input, item1_input, item2_input], 
                outputs=prediction)
    inference_model = Model(inputs=[user_input, item1_input],
                            outputs=prediction_inference)

    print(model.summary())
    return model,inference_model

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
        model_out_file = '../../model/GMF_pw_%d_%d.h5' %(num_factors, time())
        
        # Loading data
        t1 = time()
        
        dataset = Dataset(data_path,prep_data=prep_data)
        trainData, testData = dataset.train_data, dataset.testData
        num_users, num_items = dataset.num_users, dataset.num_items
        
        # Build model
        #model,inference_model = get_gmf_pairwise_model()
        model,model_inference = get_gmf_pairwise_model(num_users, num_items, num_factors)
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
        def evaulate_inference(_data):
            (hits, ndcgs, aucs) = evaluate_model_inference(model_inference, _data, dataset, topK, evaluation_threads)
            return np.array(hits).mean(), np.array(ndcgs).mean(), np.array(aucs).mean()
        
        # Init performance
        t1 = time()
        hr, ndcg, auc = evaulate(dataset.testData)
        print('Init Test: HR = %.4f, NDCG = %.4f, AUC = %.4f\t [%.1f s]' % (hr, ndcg, auc, time()-t1))
        
        _t = dataset.train_data
        user_input, item1_input,item2_input, labels = _t.pw_userids, _t.pw_posids, _t.pw_negids, np.full(len(_t.pw_userids),1.0)
        
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
                hr, ndcg, auc = evaulate_inference(dataset.testData)
                print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, AUC = %.4f [%.1f s] (inference)' 
                      % (self.epoch,  t2-t1, hr, ndcg, auc, time()-t2))
                if hr > self.best_hr:
                    self.best_hr, self.best_ndcg, self.best_iter = hr, ndcg, self.epoch
                    model.save(model_out_file, overwrite=True)
                    model_inference.save(model_out_file+".1", overwrite=True)
                self.epoch+=1
        
        t1 = time()
        metricsClbk = MetricsCallback()
        # Training
        hist = model.fit([user_input, item1_input, item2_input], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=num_epochs, verbose=2, shuffle=True, callbacks=[metricsClbk])
        
            # Evaluation
                
        model = load_model(model_out_file)
        model_inference = load_model(model_out_file+".1")
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
        hr, ndcg, auc = evaulate_inference(dataset.testData)
        print('Test [%.1f s]: HR = %.4f, NDCG = %.4f, AUC = %.4f, [%.1f s] (inference)' 
              % (t2-t1, hr, ndcg, auc, time()-t2))
        t2 = time()
        hr, ndcg, auc = evaulate_inference(dataset.testColdStart)
        print('Cold Start [%.1f s]: HR = %.4f, NDCG = %.4f, AUC = %.4f, [%.1f s] (inference)' 
              % (t2-t1, hr, ndcg, auc, time()-t2))
        t2 = time()
        hr, ndcg, auc = evaulate_inference(dataset.testColdStartPseudo)
        print('Cold Start Pseudo [%.1f s]: HR = %.4f, NDCG = %.4f, AUC = %.4f, [%.1f s] (inference)' 
              % (t2-t1, hr, ndcg, auc, time()-t2))
        
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
        prep_data=False
    )

