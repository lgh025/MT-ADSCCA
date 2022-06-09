#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#NO RISK,epochs =1000, c_index=0.7047503219344684 ？ remove log_var[i] seems as no difference.
"""
Demonstrates how the partial likelihood from a Cox proportional hazards
model can be used in a NN loss function. An example shows how a NN with
one linear-activation layer and the (negative) log partial likelihood as
loss function produces approximately the same predictor weights as a Cox
model fit in a more conventional way.
"""
import datetime
import pandas as pd
import numpy as np
import keras
#from lifelines import CoxPHFitter
#from lifelines.datasets import load_kidney_transplant

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import theano
from keras.layers import Dropout, Activation, Lambda
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Input,Embedding

from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.wrappers.scikit_learn import KerasRegressor,KerasClassifier
from lifelines.utils import concordance_index
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras.layers.core import Reshape
from keras.utils.vis_utils import plot_model
from keras.initializers import Constant
from keras.layers import  Layer
from sklearn.preprocessing import minmax_scale
import tensorflow.compat.v1 as tf1

import keras.backend as K
from keras.callbacks import LearningRateScheduler

tf1.disable_v2_behavior()
##############################

kidtx = pd.read_csv('BRCA_data_mm_name_dscca_p001_8-9_20210910.csv')
dataX01 = kidtx.drop(["Unnamed: 0","X.1","V1", "erged_data33"], axis = 1)
dataX1 = kidtx.drop(["Unnamed: 0","X.1","V1", "erged_data33"], axis = 1).values

y = np.transpose(np.array((kidtx["V1"], kidtx["erged_data33"]))) # V1=time; erged_data33=status


ytime=y[:,0:1]
ystatus= y[:,1:2] 

[ m0,n0] = dataX1.shape
dataX = np.asarray(dataX1)

dataX =minmax_scale(dataX )
data_methylation=dataX
data_mRNA=dataX
  
[ m,n] = dataX.shape
[ m1,n1] = data_methylation.shape
[ m2,n2] = data_mRNA.shape

 
dataX = dataX.reshape(m,1,n)
x=dataX
data_methylation = data_methylation.reshape(m1,1,n1)
data_mRNA = data_mRNA.reshape(m2,1,n2)

from keras.utils import np_utils
ystatus2= np_utils.to_categorical(ystatus)



##################################################################################################





def neg_log_pl(y_true, y_pred):
    # Sort by survival time (descending) so that
    # - If there are no tied survival times, the risk set
    #   for event i is individuals 0 through i
    # - If there are ties, and time[i - k] through time[i]
    #   represent all times equal to time[i], then the risk set
    #   for events i - k through i is individuals 0 through i
    sorting = tf.nn.top_k(y_true[:, 0], k =int(k_n.get_value()))
    time = K.gather(y_true[:, 0], indices = sorting.indices)
    xbeta = K.gather(y_pred[:, 0], indices = sorting.indices) #tf.gather()用来取出tensor中指定索引位置的元素。
    risk = K.exp(xbeta)

    event = K.gather(y_true[:, 1], indices = sorting.indices)
    denom = K.cumsum(risk) #这个函数的功能是返回给定axis上的累计和
    terms = xbeta - K.log(denom)
    loglik = K.cast(event, dtype = terms.dtype) * terms   #cast将x的数据格式转化成dtype

    
    
    
    
#    return -(loglik)
    return -K.sum(loglik)



## C_index metric function

def c_index3(month,risk, status):

    c_index = concordance_index(np.reshape(month, -1), -np.reshape(risk, -1), np.reshape(status, -1))

    return c_index#def get_bi_lstm_model():  

###########################################################################################

    

#      
###############################################################################################
def unique_set(Y_hazard):

    a1 = Y_hazard#.numpy()
#    print('Y_hazard:',Y_hazard)
    # Get unique times
    t, idx = np.unique(a1, return_inverse=True)

    # Get indexes of sorted array
    sort_idx = np.argsort(a1)
#    print(sort_idx)
    # Sort the array using the index
    a_sorted =a1[sort_idx]# a1[np.int(sort_idx)]# a[tf.to_int32(sort_idx)]#
#    print('a_sorted:', a_sorted)
    # Find duplicates and make them 0
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))

    # Difference a[n+1] - a[n] of non zero indexes (Gives index ranges of patients with same timesteps)
    unq_count = np.diff(np.nonzero(unq_first)[0])

    # Split all index from single array to multiple arrays where each contains all indexes having same timestep
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))

    return t, unq_idx

###########################################################################################

###################################################################################################


################################################################################################### 

###########################################################################################        
  
#########################################################################################################################

   
#############################################################################################################
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0:# and epoch != 0:
        lr2=[0.0005,0.0004,0.0004,0.0002,0.0001,0.00005,0.00004,0.00002,0.00001,0.000005,0.000002]
        learning_rate=lr2[int(epoch/100)]
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, (lr/lr )* learning_rate)
        print("lr changed to {}".format((lr/lr )* learning_rate))
    return K.get_value(model.optimizer.lr)

####################################################################################################################    
seed = 63
np.random.seed(seed)
kf = KFold(n_splits=10, shuffle=True, random_state=seed)
ypred=[]
ypred_train=[]
xtest_original=[]
status_new=[]
time_new=[]
index2=[]
iFold = 0
EPOCH =1000
for train_index, val_index in kf.split(x):
    iFold = iFold+1
#    train_x, test_x, train_y, test_y,= X[train_index], X[val_index], y[train_index], y[val_index] # 这里的X_train，y_train为第iFold个fold的训练集，X_val，y_val为validation set
    x_train, x_test, y_train, y_test, ytime_train, ytime_test, ystatus_train, ystatus_test, ystatus2_train, ystatus2_test =\
        dataX[train_index], dataX[val_index], y[train_index], y[val_index], ytime[train_index], ytime[val_index], ystatus[train_index],ystatus[val_index],\
                           ystatus2[train_index],ystatus2[val_index]
    
    input_dim =x_train.shape[2]
    output_dimM = y_train.shape[1]
    output_dimA = 1
    n1 = y_train.shape[0]
    
#    k_n = theano.shared(np.asarray(n,dtype=theano.config.floatX),borrow=True)
    k_n = theano.shared(n1,borrow=True)
    k_ytime_train = theano.shared(ytime_train,borrow=True)
    k_ystatus_train = theano.shared(ystatus_train,borrow=True)
    N = theano.shared(n1,borrow=True)
    R_matrix = np.zeros([n1, n1], dtype=int)
    R_matrix =theano.shared(R_matrix,borrow=True)
##############################################3    
    
    Y_hazard0=y_train[:,0]
    Y_survival=y_train[:,1]

#    Y_hazard1M=tf.reshape(Y_hazard0, [-1, Y_train.shape[0],1])
#    Y_hazard=Y_hazard1M[-1,:,-1]
    
    t0, H0 = unique_set(Y_hazard0) # t:unique time. H original index.
    
    actual_event_index = np.nonzero(Y_survival)[0]
    H0 = [list(set(h) & set(actual_event_index)) for h in H0]
    ordinal_n = np.asarray([len(h) for h in H0])
    Hj=sum(H0[0:],[])
    
    k_ordinal_H = theano.shared(np.asarray(Hj),borrow=True)
    k_ordinal_t = theano.shared(t0,borrow=True)
    k_ordinal_n = theano.shared(ordinal_n,borrow=True)
 #########################################################################################################################   

    class varHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.lambdas0 = []
            self.lambdas1 = []
        def on_epoch_end(self, batch, logs={}):
            self.lambdas0.append( K.get_value(self.model.layers[-1].log_vars[0]))
            self.lambdas1.append( K.get_value(self.model.layers[-1].log_vars[1]))
 #############################################################################################################################    
#    input_dim0 =theano.shared(input_dim,borrow=True)
# Build model structure
    # gene Only
    gene_input = Input(name='gene_input', shape=(1,input_dim))
#    out1=Bidirectional(LSTM(55,activation='linear',return_sequences=True,kernel_initializer=glorot_uniform(),kernel_regularizer=l2(reg),activity_regularizer=l2(0.001)), merge_mode='concat')(title_input)
#    out_gene=Bidirectional(LSTM(55,return_sequences=False), merge_mode='concat')(gene_input)
    
    out_gene=Bidirectional(LSTM(55,activation='tanh', return_sequences=False,\
        kernel_initializer=glorot_uniform(),kernel_regularizer=l2(0.0005),activity_regularizer=l2(0.001)),  merge_mode='concat')(gene_input)
    main_output= Dense(1,activation='linear',name='main_output')(out_gene)

    model = Model(inputs=[gene_input],outputs=[main_output])
    model.summary()
    
    model.compile(optimizer='adam', loss=neg_log_pl)
    

    
    
    reduce_lr = LearningRateScheduler(scheduler)
    Lambdas = varHistory() 
    hist = model.fit(x_train, y_train, batch_size = n1, epochs =EPOCH,callbacks=[reduce_lr])
    

    
    plt.plot(range(EPOCH), hist.history['loss'])
    plt.title('Training Loss Curves only main loss')
    plt.ylabel('train-loss Value')
    plt.xlabel('epoch')
#    plt.legend(['${lambda0}$','${lambda1}$'], loc='upper right')
    plt.savefig('Loss Curves only main loss316.jpg', dpi=300) #指定分辨率保存
    plt.show()
    
    
    np.savetxt("hist.history['loss1000_ml_ordCOX20201220']_without_penalty_term_only_main_loss316.csv",hist.history['loss'], delimiter=",")
    

    prediction = model.predict(x_test)
#    prediction =predicted_main+0*predicted_aux
    
    c_index2=c_index3( np.asarray(ytime_test),np.asarray(prediction), np.asarray(ystatus_test))
    
    print( c_index2)
    
#############################################################################################################################    
   
    ypred.extend(prediction)
#    ypred_train.extend(prediction_train_median)
#    xtest_original.extend(x_test)
    index2.extend(val_index)
    status_new.extend(ystatus[val_index])
    time_new.extend(ytime[val_index])
#    print(ypred.shape)
    
    K.clear_session()
    tf1.reset_default_graph()
    print(iFold)
    nowTime = datetime.datetime.now()
    print("nowTime: ",nowTime)
np.savetxt("brca_prediction1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_mainloss316.csv", ypred, delimiter=",")
np.savetxt("brca_ytime_test1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_mainloss316.csv", time_new, delimiter=",")
np.savetxt("brca_ystatus_test1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_mainloss316.csv", status_new, delimiter=",")
np.savetxt("brca_ypred_train_median1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_mainloss316.csv", ypred_train, delimiter=",")

df = pd.read_csv("brca_prediction1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_mainloss316.csv",header=None)    
month=np.asarray(pd.read_csv("brca_ytime_test1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_mainloss316.csv",header=None)) 
status=np.asarray(pd.read_csv("brca_ystatus_test1204_18lstm2222_epoch400_drop01_resnet_without penalty term_only_mainloss316.csv",header=None)) 




risk=np.asarray(df)
c_indices_only_main = c_index3(month, risk,status)
print("c_indices:",c_indices_only_main)
#np.savetxt("c_indices_nn827.txt", c_indices_mlp, delimiter=",")
np.save("c_indices_without penalty term_only_mainloss316",c_indices_only_main) 
print("brca_methylation_mRNA_lmqcm_lr_backend2233_only_main_loss20210316.py") 
print("brca only_main:dataX=data_rna+methylation:",c_indices_only_main)
data_a=np.load('c_indices_without penalty term_only_mainloss316.npy')

aa=0