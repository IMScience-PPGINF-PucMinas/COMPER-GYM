from statistics import mode
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from pathlib import Path
import os
import gc


class RNN(object):
    def __init__(self,inputshapex=1,inputshapey=35,output_dim=1,batch_size=128,verbose=True,netparamsdir='./',optimizer='rmsprop',early_stopping=False):
        self.paramsidr = netparamsdir
        self.verbose =verbose
        self.optimizer = optimizer
        self.outputdim = output_dim
        self.rms_prop_optimizer =RMSprop(learning_rate=0.001) #tf.train.RMSPropOptimizer(learning_rate=0.00025)
        self.early_stopping_callback=None
        self.model_checkpoint_callback = None
        self.checkoint_path = "./"
        self.early_stopping =False
        self.checking_point = False
        self.inputshapex = inputshapex
        self.inputshapey = inputshapey


        
        self.create_lstm()
        if(early_stopping):
            self.config_early_stopping()           
             
    
    def create_lstm(self):
        self.lstm = Sequential()                
        self.lstm.add(LSTM(128,return_sequences=True,stateful=False,input_shape=(self.inputshapex,self.inputshapey),activation='tanh'))
        self.lstm.add(LSTM(128,return_sequences=True,activation='tanh'))
        self.lstm.add(LSTM(128,activation='tanh'))              
        self.lstm.add(Dense(1))

         

    def compile(self,reload_weights_if_exists=True): #adam
        loaded = self.load_weights() 
        if(not loaded):
            loaded = self.load_states()       
        if(self.optimizer=='rmsprop'):
            self.lstm.compile(loss="mean_squared_error",optimizer=self.rms_prop_optimizer)
        else:
            self.lstm.compile(loss="mean_squared_error",optimizer='adam')

    def predict(self,x):
        #return self.lstm(x).numpy()
        return self.lstm(x).numpy()


    def fit(self,x=None, y=None, batch_size=None, epochs=1, verbose=0, callbacks=None, validation_split=0., validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None):
        if(callbacks==None and self.early_stopping):
            callbacks=[]
            callbacks.append(self.early_stopping_callback)
            
        return self.lstm.fit(x,y,batch_size,epochs,verbose,callbacks,validation_split,validation_data,shuffle,class_weight,sample_weight,initial_epoch,steps_per_epoch,validation_steps)

    def save_weights(self):
        os.makedirs(self.paramsidr, exist_ok=True)
        paramdir = self.paramsidr + '/qlstm_weights.h5'
        self.lstm.save_weights(paramdir)
    def save_states(self):
        os.makedirs(self.paramsidr, exist_ok=True)
        paramdir = self.paramsidr + '/qlstm_states.h5'
        self.lstm.save(paramdir)
    def load_weights(self):
        loaded= False
        print("\nQLSTM TRY WEIGHTS LOADING!\n")
        paramdir = self.paramsidr + '/qlstm_weights.h5'
        if(os.path.exists(paramdir)):
            self.lstm.load_weights(paramdir)
            print("\nQLSTM WEIGHTS LOADED!\n")
            loaded = True
        return loaded

    def load_states(self):
        print("\nQLSTM TRY STATES LOADING!\n")
        loaded= False        
        paramdir = self.paramsidr + '/qlstm_states.h5'
        if(os.path.exists(paramdir)):
            self.lstm = keras.models.load_model(paramdir)
            print("\nQLSTM STATES LOADED!\n")
            loaded = True
        return loaded
    
    def set_set_checkpoint_path(self,path):
        self.checkoint_path = path

    def config_early_stopping(self,monitor="loss",patience=50):
        self.early_stopping_callback = keras.callbacks.EarlyStopping(monitor=monitor,mode='min',verbose=1,patience=patience)
        self.early_stopping =True
    
    def config_checkpoint(self):
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
        self.checkpoint_path=self.checkpoint_path+self.name+"_checkpoint.h5"
        self.modelckpt_callback = keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode='min',
            filepath=self.checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
        )
        self.checking_point = True

