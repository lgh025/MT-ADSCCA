#!/usr/bin/env python3
# -*- coding: utf-8 -*-
           
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
from keras.layers import  concatenate
#from cca_layer import CCA
import keras.backend as K
from keras.callbacks import LearningRateScheduler
#from keras_custom import cca_loss, batch_cca_loss,  PickMax, UnitNormWithNonnneg
from keras.regularizers import l1
tf1.disable_v2_behavior()

##############################
##############################

kidtx = pd.read_csv('brca__methylation_mRNA_raw_data.csv')



dataX01 = kidtx.drop(["Unnamed: 0","X.1","V1", "erged_data33","X"], axis = 1)
dataX1 = kidtx.drop(["Unnamed: 0","X.1","V1", "erged_data33","X"], axis = 1).values

y = np.transpose(np.array((kidtx["V1"], kidtx["erged_data33"]))) # V1=time; erged_data33=status

[ m0,n0] = dataX1.shape

col_name=list(dataX01)
y_name=["V1", "erged_data33"]

col_name_meth=col_name[0:679]
col_name_mRNA=col_name[679:n0]

dataX = np.asarray(dataX1)
dataX =minmax_scale(dataX ) 
data_methylation=dataX[:,0:679]
data_mRNA=dataX[:,679:n0]

[ m,n] = dataX.shape
[ m1,n1] = data_methylation.shape
[ m2,n2] = data_mRNA.shape

 
#dataX = dataX.reshape(m,1,n)
x=dataX
#data_methylation = data_methylation.reshape(m1,1,n1)
#data_mRNA = data_mRNA.reshape(m2,1,n2)

ytime=np.transpose(np.array(kidtx["V1"])) # only V1=time;
ystatus= np.transpose(np.array(kidtx["erged_data33"])) #only erged_data33=status

from keras.utils import np_utils
ystatus2= np_utils.to_categorical(ystatus)



##################################################################################################


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
###########################################################################################

############################################################################################

################################################################################################### 

def py_func(func, inp, Tout, stateful=True, name=None, grad_func=None):

    grad_name = 'PyFuncGrad_' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(grad_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        func1=tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        return func1
###########################################################################################        

###########################################################################################        
    
#########################################################################################################################

#############    #######################################################################

import theano.tensor
import theano.tensor.nlinalg as TN
import theano.tensor.basic as TB


#######################################################################################3
def scca_loss(x_proj,y_proj,u,v):
#   
    x1tx2 = tf.abs(tf.matmul(tf.transpose(x_proj), y_proj))
    ux1tx2 = tf.matmul(tf.transpose(u), x1tx2)
    ux1tx2v =tf.matmul(ux1tx2, v)
#    cca_loss2 = -ux1tx2v
    
    xu_yv=tf.matmul(x_proj, u)-tf.matmul(y_proj, v)
    omega=tf.reduce_sum(tf.norm(xu_yv,ord=2))
    cca_loss2 = -ux1tx2v/(2*omega*omega)
    
    clipped_u = tf.clip_by_norm(u, clip_norm=1.0, axes=0)
    clip_u = tf.assign(u, clipped_u, name='ortho')
    tf.add_to_collection('normalize_ops', clip_u)
    clipped_v = tf.clip_by_norm(v, clip_norm=1.0, axes=0)
    clip_v = tf.assign(v, clipped_v, name='ortho')
    tf.add_to_collection('normalize_ops', clip_v)
    
    ## l1 penalty
    l1_u =tf.reduce_sum(tf.abs(u))
    l1_v =tf.reduce_sum(tf.abs(v))
    
    ## total loss
    total_loss = cca_loss2# + l1_u * L1_PENALTY + l1_v * L1_PENALTY
    return total_loss

def scca_loss_u2(x_proj,y_proj,u,v):

    
    
   # u2=X1'*(x2*v);
    x2v = tf.matmul(y_proj, v)
    u2=tf.matmul(tf.transpose(x_proj), x2v)
    ## l1 penalty
    l1_u =tf.reduce_sum(tf.abs(u))
#    l1_v =tf.reduce_sum(tf.abs(v))
    
    l2u=tf.reduce_sum(tf.norm(u2,ord=2))
    ## total loss
#    total_loss = cca_loss2# + l1_u * L1_PENALTY + l1_v * L1_PENALTY
    return l1_u#+l2u

def scca_loss_v2(x_proj,y_proj,u,v):

    
    x1u = tf.matmul(x_proj, u)
    v2=tf.matmul(tf.transpose(x_proj), x1u)
    ## l1 penalty
   # l1_u =tf.reduce_sum(tf.abs(u))
    l1_v =tf.reduce_sum(tf.abs(v))
    l2v=tf.reduce_sum(tf.norm(v2,ord=2))
    ## total loss
  #  total_loss = cca_loss2# + l1_u * L1_PENALTY + l1_v * L1_PENALTY
    return l1_v#+l2v
    

    
def scca_loss_zz(x_proj,y_proj,u,v,z):
    xu=tf.matmul(x_proj, u)
    xu_z=xu-z
    l3_1=tf.reduce_sum(tf.abs( xu_z))
    l3_2=tf.reduce_sum(tf.norm( xu_z,ord=2))/2
    
    
    return l3_1+l3_2

def scca_loss_hh(x_proj,y_proj,u,v,h):
    yv=tf.matmul(y_proj, v)
    yv_h=yv-h
    l4_1=tf.reduce_sum(tf.abs( yv_h))
    l4_2=tf.reduce_sum(tf.norm( yv_h,ord=2))/2
    return l4_1+l4_2


def L1_loss(x_proj,y_proj):
   

    covar_mat = tf.matmul(tf.transpose(x_proj), y_proj)
    diag_sum = tf.reduce_sum(tf.abs(tf.diag_part(covar_mat)))
    cca_score = tf.multiply(-1., diag_sum)
    #inter_sum = tf.reduce_sum(tf.abs(tf.matrix_band_part(covar_mat, 0, -1)))
    #cca_score = tf.multiply(-1., diag_sum - inter_sum) 
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    losses = tf.add_n(reg_losses)
    return losses  #total_loss     
    
#########################################################################################################################

   
#############################################################################################################  
#########################################################################################################################
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # initialise log_vars
        self.log_vars = []
#        self.u = []
#        self.log_vars = []
#        u = K.get_variable('u', shape=(ys_true.shape[-1], 1))
#        v = K.get_variable('v', shape=(ys_pred.shape[-1], 1))
        
        self.uu=self.add_weight(name='u', shape=(input_shape[0][-1], 1),  initializer='random_normal', trainable=True)
        self.vv=self.add_weight(name='v', shape=(input_shape[1][-1], 1),  initializer='random_normal', trainable=True)
        self.zz=self.add_weight(name='zz', shape=(int(k_n.get_value()), 1),  initializer='random_normal', trainable=True)
        self.hh=self.add_weight(name='hh', shape=(int(k_n.get_value()), 1),  initializer='random_normal', trainable=True)
        for i in range(self.nb_outputs+3):
           
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
#        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        log_var=self.log_vars
        uu=self.uu
        vv=self.vv
        zz=self.zz
        hh=self.hh
#        for y_true, y_pred, log_var ,ii in zip(ys_true, ys_pred, self.log_vars,self.nb_outputs):
        for i in range(self.nb_outputs+3):    
            precision = (K.exp(-log_var[i]))#**0.5
#            precision = log_var[i]#**0.5
            #precision= tf.clip_by_value(precision, 0., 1.)
            if i==0:
#                lossA=neg_log_pl(ys_true[i], ys_pred[i])
                
                lossA=scca_loss(ys_true, ys_pred,uu,vv)
#                lossA=scca_loss(ys_true, ys_pred)
                loss += K.sum( lossA + log_var[i], -1)
#                print("Neg_loss:",lossA.item())
            if i==1: 
#                 lossA = py_func(ordinal_loss,
#                     [ys_true[i-1], ys_pred[i]], [tf.float32],
#                     name = "ordinal_loss",
#                     grad_func = ordinal_loss_grad )[0]
#                 lossA.set_shape((ys_true[i].shape[0],))
#                lossA=LOSS_L2(ys_true[i], ys_pred[i])
                lossA=scca_loss_u2(ys_true, ys_pred,uu,vv)
                loss += K.sum(precision * lossA+ log_var[i], -1)
                
            if i==2: 
#                 lossA = py_func(ordinal_loss,
#                     [ys_true[i-1], ys_pred[i]], [tf.float32],
#                     name = "ordinal_loss",
#                     grad_func = ordinal_loss_grad )[0]
#                 lossA.set_shape((ys_true[i].shape[0],))
#                lossA=LOSS_L2(ys_true[i], ys_pred[i])
                lossA=scca_loss_v2(ys_true, ys_pred,uu,vv)
                loss += K.sum(precision * lossA+ log_var[i], -1)
                
            if i==3: 
                lossA=scca_loss_zz(ys_true, ys_pred,uu,vv,zz)
                loss += K.sum(precision * lossA+ log_var[i], -1)          
            if i==4: 
                lossA=scca_loss_hh(ys_true, ys_pred,uu,vv,hh)
                loss += K.sum(precision * lossA+ log_var[i], -1)      
                
#                lossA=neg_log_pl(ys_true[i-1], ys_pred[i])
#                print("Ordinal_loss:",lossA.item())
#                lossA=tf.py_function(func=ordinal_loss, inp=[ys_true[i], ys_pred[i]], Tout=tf.float32) 
#                lossA=ordinal_loss(Y_trueM, model_aux(tf.to_float(x_train0M)))
#                lossA=neg_log_pl_1(ys_true[i-1], ys_pred[i])
#            loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
#            loss += K.sum(precision * lossA, -1)
#            loss += K.sum(precision * lossA + log_var[i], -1)
        
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs-1][-1]
        ys_pred = inputs[self.nb_outputs-1:][-1]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)
    #######################################################################
   
#############################################################################################################       
#############################################################################################################
def scheduler2(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0:# and epoch != 0:
        lr2=[0.0005,0.0004,0.0004,0.0002,0.0001,0.00005,0.00004,0.00002,0.00001,0.000005,0.000002]
        #lr2=[0.0005,0.0001,0.00005,0.00001,0.000005,0.000005,0.000005,0.000005,0.000005,0.000005,0.000005]
        learning_rate=lr2[int(epoch/100)]
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, (lr/lr )* learning_rate)
        print("lr changed to {}".format((lr/lr )* learning_rate))
    return K.get_value(model.optimizer.lr)

####################################################################################################################       
    
    
    
    
    
    
#############################################################################################################
def scheduler22(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0:# and epoch != 0:
        lr2=[0.005,0.004,0.004,0.002,0.001,0.0005,0.0004,0.0002,0.0001,0.00005,0.00002]
        learning_rate=lr2[int(epoch/100)]
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, (lr/lr )* learning_rate)
        print("lr changed to {}".format((lr/lr )* learning_rate))
    return K.get_value(model.optimizer.lr)

####################################################################################################################    
sparsity=(1e-5,1e-5)
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
W_uu=[]
W_vv=[]
for train_index, val_index in kf.split(x):
    iFold = iFold+1
#    train_x, test_x, train_y, test_y,= X[train_index], X[val_index], y[train_index], y[val_index] # 这里的X_train，y_train为第iFold个fold的训练集，X_val，y_val为validation set
    x_train, x_test, y_train, y_test, ytime_train, ytime_test, ystatus_train, ystatus_test,data_methylation_train,data_methylation_test,data_mRNA_train,data_mRNA_test, ystatus2_train, ystatus2_test =\
        dataX[train_index], dataX[val_index], y[train_index], y[val_index], ytime[train_index], ytime[val_index], ystatus[train_index],ystatus[val_index],\
        data_methylation[train_index],data_methylation[val_index],data_mRNA[train_index],data_mRNA[val_index],ystatus2[train_index],ystatus2[val_index]
    
#    input_dim =x_train.shape[2]
#    output_dimM = y_train.shape[1]
#    output_dimA = 1
#    n1 = y_train.shape[0]
    input_dim =x_train.shape[1]
    input_dim_mRNA =data_mRNA.shape[1]
#    ytrue_dim=y_train.shape[1]
    input_dim_methylation =data_methylation.shape[1]
    
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

    
#    #######################################################################################
    class varHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.lambdas0 = []
            self.lambdas1 = []
            self.lambdas2 = []
            self.lambdas3 = []
            self.u = []
            self.v = []
        def on_epoch_end(self, batch, logs={}):
            self.lambdas0.append( K.get_value(self.model.layers[-1].log_vars[1]))
            self.lambdas1.append( K.get_value(self.model.layers[-1].log_vars[2]))
            self.lambdas2.append( K.get_value(self.model.layers[-1].log_vars[3]))
            self.lambdas3.append( K.get_value(self.model.layers[-1].log_vars[4]))
            self.uu= K.get_value(self.model.layers[-1].uu)
            self.vv= K.get_value(self.model.layers[-1].vv)
            self.zz= K.get_value(self.model.layers[-1].zz)
            self.hh= K.get_value(self.model.layers[-1].hh)
     #############################################################################################################################    
    
    #    
    ###############################################################################################################################################    
     ####################
    # X projection model
    ####################
#    modelX = Sequential()
    gene_input = Input(name='gene_input', shape=[input_dim_mRNA])
    out_gene=Dense(10, bias=False, 
                 W_constraint=UnitNormWithNonnneg(False),
                 W_regularizer=l1(sparsity[0]),
                 name='main_output')(gene_input)

 
    methylation_input = Input(name='methylation_input', shape=[input_dim_methylation])

    out_methylation=Dense(10, bias=False,
        W_constraint=UnitNormWithNonnneg(False),
        W_regularizer=l1(sparsity[1]), name='aux_output')( methylation_input) #activation='linear',
    
    
#########################################################################################################################################################

#####################################################################################################################################################     

#    
    
    y1_true = Input(shape=(2,), name='y1_true')
#    y1_true = Input(shape=(output_dimM,), name='y1_true')
    y2_true = Input(shape=(2,), name='y2_true')
#    out = CustomMultiLossLayer2(nb_outputs=2)([out_gene, out_methylation])
    out = CustomMultiLossLayer(nb_outputs=2)([gene_input, methylation_input])
    model =Model([gene_input,methylation_input], out)
    model.summary()
    model.compile(optimizer='adam', loss=None)
    
    
 #####################################################################################################################################################     

    
    
    reduce_lr = LearningRateScheduler(scheduler2)
    Lambdas = varHistory() 
    hist = model.fit([data_mRNA_train,data_methylation_train], batch_size = n1, epochs =EPOCH,callbacks=[reduce_lr,Lambdas])
    
    
    lambda0=np.asarray( Lambdas.lambdas0)
    lambda1=np.asarray( Lambdas.lambdas1)
    lambda2=np.asarray( Lambdas.lambdas2)
    lambda3=np.asarray( Lambdas.lambdas3)
    
    uu=np.asarray( Lambdas.uu)
    vv=np.asarray( Lambdas.vv)
    zz=np.asarray( Lambdas.zz)
    hh=np.asarray( Lambdas.hh)
    
    std0=(np.exp(-lambda0))#**0.5
    std1=(np.exp(-lambda1))#**0.5
    std2=(np.exp(-lambda2))#**0.5
    std3=(np.exp(-lambda3))#**0.5
    print("iFold=",iFold)
    plt.plot(range(EPOCH), lambda0,color='red')
#    plt.show() 
    plt.plot(range(EPOCH), lambda1,color='purple')
    plt.plot(range(EPOCH), lambda2,color='blue')
    plt.plot(range(EPOCH), lambda3,color='black')
    
    plt.title('MT-ADSCCA Lambda Curves')
    plt.ylabel('Train-lambda Value')
    plt.xlabel('epoch')
    plt.legend(['lambda1','lambda2','lambda3','lambda4'], loc='center right')
    plt.savefig('MT-ADSCCA Lambda Curves1234.jpg', dpi=300) #指定分辨率保存
    plt.show()
    
    plt.plot(range(EPOCH), std0,color='red')
#    plt.show() 
    plt.plot(range(EPOCH), std1,color='purple')
    plt.plot(range(EPOCH), std2,color='blue')
    plt.plot(range(EPOCH), std3,color='black')
    plt.title('MT-ADSCCA Weight Curves')
    plt.ylabel('Train-weight Value')
    plt.xlabel('epoch')
    plt.legend(['lambda1','lambda2','lambda3','lambda4'], loc='upper right')
    plt.savefig('MT-ADSCCA lambda1-lambda4 Curves term315.jpg', dpi=300) #指定分辨率保存
    plt.show()
    
    plt.plot(range(EPOCH), hist.history['loss'])
    plt.title('Training Loss Curves MT-ADSCCA')
    plt.ylabel('train-loss Value')
    plt.xlabel('epoch')
#    plt.legend(['${lambda0}$','${lambda1}$'], loc='upper right')
    plt.savefig('Loss Curves without penalty term315.jpg', dpi=300) #指定分辨率保存
    plt.show()
    
    
    np.savetxt("BRCA_methylation_mRNA_MT-ADSCCA_20210902.csv",hist.history['loss'], delimiter=",")
    
#    import pylab
#    pylab.plot(hist.history['loss'])
#    print([np.exp(-K.get_value(log_var[0]))**0.5 for log_var in model.layers[-1].log_vars])
    print('lambda1 for Main Loss=',lambda0[999])
    print('lambda2 for Auxiliary Loss=',lambda1[999])
    ###########
#    print('weight1 for Main Loss=',std0[999])
    print('weight1 for Auxiliary1 Loss=',std0[999])
    print('weight2 for Auxiliary Loss=',std1[999])
    print('weight3 for Auxiliary Loss=',std2[999])
    print('weight4 for Auxiliary Loss=',std3[999])
    
    print('uu for data_gene=',uu)
    print('vv for methylation data=',vv)
    ###############
    idx1=(np.abs(uu)>0.01)[:,-1]
    print('idx1 for data_gene=',idx1)
    W_u=np.arange(input_dim_mRNA)
    W1=W_u[idx1]
    W_uu.extend(W_u[idx1])
    
    idx2=(np.abs(vv)>0.01)[:,-1]
    print('idx2 for data_gene=',idx2)
    W_v=np.arange(input_dim_methylation)
    W2=W_v[idx2]
    W_vv.extend(W_v[idx2])
    print("iFold2=",iFold)
   ##########################################################################################
##########################################################################################
 ##########################################################################################
Au= pd.value_counts(W_uu)
Av=pd.value_counts(W_vv)

Gr_index=Au>=8
Gm_index=Av>=9

idx_r=np.array((Gr_index[Gr_index]).index.tolist())
idx_m=np.array((Gm_index[Gm_index]).index.tolist())

mRNA_names3=[col_name_mRNA[idx_r2] for idx_r2 in idx_r]
meth_names3=[col_name_meth[idx_m2] for idx_m2 in idx_m]


data_mRNA_train3=data_mRNA[:,Gr_index]
data_mRNA_train4=pd.DataFrame(data_mRNA_train3,columns=mRNA_names3)

data_methylation_train3=data_methylation[:,Gm_index]
data_methylation_train4=pd.DataFrame(data_methylation_train3,columns=meth_names3)

data_mm=np.concatenate([data_mRNA_train3,data_methylation_train3],1)
data_mm2=np.concatenate([y,data_mm],1)

data_mm_name4=pd.concat([data_mRNA_train4,data_methylation_train4],1)

y_data_name=pd.DataFrame(y,columns=y_name)

data_y_mm_name55=pd.concat([kidtx["X.1"],y_data_name,data_mm_name4],1) ############ add colomn name
#data_mRNA_test3=data_mRNA_test[:,Gr_index]
#data_methylation_test3=data_methylation_test[:,Gm_index]
#data_mm_test=np.concatenate([data_mRNA_test3,data_methylation_test3],1)

[ m3,n3] = data_mm.shape
#[ m4,n4] = data_mm_test.shape
   

 
#data_mm3 = data_mm.reshape(m3,1,n3)
#data_mm_test3=data_mm_test.reshape(m4,1,n4)

#np.savetxt("BRCA_data_mm_dscca_p001_8-9_20210910.csv",data_mm2, delimiter=",")

#pd.dataFrame.to_save()
data_y_mm_name55.to_csv("BRCA_data_mm_name_dscca_p001_8-9_20210910.csv")


aaaaaaaaaaaaaaaaaa=0
 