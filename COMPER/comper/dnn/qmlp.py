import numpy as np
import keras
from keras.models import Sequential,Model,Input
from keras.layers import Dense, Dropout, Flatten,Activation,Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import keras.backend as K
import os
from comper.config.transitions import FrameTransition as ft
from comper.config.transitions import FrameTransitionTypes as ft_types
from comper.config import parameters as param
from comper.config.exceptions import ExceptionRunType
#sess = tf.Session()
#K.set_session(sess)
class MlpNet(object):
    def __init__(self,netparamsdir='./',run_type=param.RunType.TRAIN):
        self.paramsidr = netparamsdir
        self.run_type = run_type
        self.mlp = {}
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    
    def create(self,input_shape,output_dim=3):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.mlp = tf.keras.Sequential([            
            tf.keras.layers.Dense(256,input_shape=input_shape,name="Dense1"),            
            tf.keras.layers.Dense(256,name="Dense2"),
            tf.keras.layers.Flatten(name="Flatten"),
            tf.keras.layers.Dense(output_dim,activation=None,name="DenseOut",kernel_initializer=last_init)
        ])   

    def compile_model(self):
        try:
            loaded = self.load_weights()
            if(not loaded):
                loaded = self.load_states()
            if(not loaded and self.run_type==param.RunType.TEST):
                raise ExceptionRunType(self.run_type,message="Weigths not found to run on test mode.")
        except Exception as isnt:
            print(type(isnt),isnt) 
            pass           
            
        #self.mlp.compile(loss=keras.losses.mean_squared_error,optimizer=self.optimizer)
    
    def get_loss(self,states,qtargets):
        q = tf.reduce_sum(self.mlp(states),axis=1)
        #loss = tf.keras.losses.mean_squared_error(q,qtargets)
        loss = tf.compat.v1.losses.huber_loss(qtargets, q, reduction=tf.compat.v1.losses.Reduction.NONE)
        return tf.reduce_mean(loss)

    def grad(self,loss):            
       gradients = self.optimizer.get_gradients(loss, self.mlp.trainable_variables) # gradient tensors      
       return gradients

    def fit_model(self,states,qtargets):        
        s = tf.compat.v1.placeholder(dtype=tf.float32,shape=states.shape)
        qt= tf.compat.v1.placeholder(dtype=tf.float32,shape=qtargets.shape)

        loss_value = self.get_loss(s,qt)
        grads = self.grad(loss_value)       
        train_op =self.optimizer.apply_gradients(zip(grads,self.mlp.trainable_variables)) 

        init_op = tf.compat.v1.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init_op)        
        sess.run(train_op,feed_dict={s:states,qt:qtargets})
        sess.close()
        #with sess.as_default():            
           #train_op.run(feed_dict={s:states,qt:qtargets})
    
    def forward(self,state):
        #stateph =  tf.placeholder(dtype=tf.float32, shape=state.shape)        
        fop = self.mlp.predict_on_batch(state)        
        #pred = np.array
        #init_op = tf.global_variables_initializer()
        #sess.run(init_op)
        #with sess.as_default():            
           #pred = sess.run(fop,feed_dict={self.inputs:state})
        #return pred
        return fop

    def save_weights(self):
        os.makedirs(self.paramsidr, exist_ok=True)
        paramdir = self.paramsidr + '/dnn_weights.h5'
        self.mlp.save_weights(paramdir)
    def save_states(self):
        os.makedirs(self.paramsidr, exist_ok=True)
        paramdir = self.paramsidr + '/dnn_states.h5'
        self.mlp.save(paramdir)
            
    def load_weights(self):
        try:
            loaded= False
            print("\nMLP TRY WEIGHTS LOADING!\n")
            paramdir = self.paramsidr + '/mlp_weights.h5'
            print(paramdir)
            if(os.path.exists(paramdir)):
                print(paramdir)
                self.mlp.load_weights(paramdir)
                print("\nMLP WEIGHTS LOADED!\n")
                loaded = True
            return loaded
        except ValueError as isnt:
            print(type(isnt),isnt)
            raise isnt

    def load_states(self):
        print("\nMLP TRY STATES LOADING!\n")
        loaded= False        
        paramdir = self.paramsidr + '/mlp_states.h5'
        if(os.path.exists(paramdir)):
            self.mlp = tf.keras.models.load_model(paramdir)
            print("\nMLP STATES LOADED!\n")
            loaded = True
        return loaded


    



              




              
