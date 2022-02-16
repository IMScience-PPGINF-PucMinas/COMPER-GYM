
import numpy as np
#from matplotlib import pyplot
import pandas as pd
from pandas import DataFrame
from pandas import concat
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from data import lstm_logger
from qrnn.rnn import RNN
from config.transitions import FrameTransition as ft


class QLSTMGSCALE(object):
    def __init__(self, transitions_memory, reduced_transitions_memory, inputshapex=1, inputshapey=35, outputdim=1, verbose=False, transition_batch_size=64, netparamsdir='./', target_optimizer='rmsprop', log_dir="",log_train=True,target_early_stopping=False):

        self.train_loss_history = list()
        self.train_val_loss_history = list()
        self.train_val_rmse_history = list()
        self.scaler = MinMaxScaler(feature_range=(0, 1))        
        self.lstm_bacth_size = 32
        self.transitions_default_batch_size = transition_batch_size
        self.transitions_real_batch_size = 0
        self.inputshapex = inputshapex
        self.inputshapey = inputshapey
        self.verbose = verbose
        self.training = True
        self.target_early_stopping = target_early_stopping
        self.transitions_memory = transitions_memory
        self.transitions_memory_l = reduced_transitions_memory
        self.netparamsdir = netparamsdir
        self.target_optimizer = target_optimizer
        self.outputdim = outputdim
        self.logDir = log_dir
        self.logTrain = log_train
        self.trainPredictionCount = 0
        self.features_to_drop = []


        self.__initialize()
        
    
    def __initialize(self):        
        self.__config_logger()
        self.__define_features_to_drop()
        self.__initialize_ltsm_q_prediction()
    
    def __config_logger(self):
        if(self.logDir != ""):
            self.logDir = self.logDir+"/qlstm"
            lstm_logger.session(self.logDir).__enter__()

    def __define_features_to_drop(self):
        start =  ft.T_IDX_Q
        end   =  ft.T_N_IDX*2           
        for f in range(start, end):            
            if(f != end -1):
                self.features_to_drop.append(f)

    def __initialize_ltsm_q_prediction(self):
        self.lstm = RNN(inputshapex=self.inputshapex, inputshapey=self.inputshapey,
                        output_dim=self.outputdim, batch_size=self.lstm_bacth_size, 
                        netparamsdir=self.netparamsdir,early_stopping=self.target_early_stopping)
        self.lstm.compile()
    
    

    def save_weights(self):
        self.lstm.save_weights()

    def save_states(self):
        self.lstm.save_states()
    
    def __load_transition_featrues_from_memory(self):

        try:
            transition_features = self.transitions_memory.load_transitions_batch_as_features_array(bsize=self.transitions_default_batch_size,normalizeFromMean=True)
            self.transitions_real_batch_size = len(transition_features)
            transformed = self.transform_transitions(transition_features, 1, 1)           
            if(self.verbose):                
                print("\n transformed", transformed.shape)
                print(transformed.head(5))
                
            transformed.drop(transformed.columns[self.features_to_drop], axis=1, inplace=True)

            if(self.verbose):                
                print("\n droped columns", transformed.shape)
                print(transformed.head(5))
            values_col=transformed.columns.values[-1]
            values = transformed[values_col].to_numpy()
            values = values.reshape(values.shape[0],1)
            transformed.drop([values_col], axis=1,inplace=True)
            transformed = transformed.to_numpy()
            
            transformed = self.scaler.fit_transform(transformed)
            transformed = np.concatenate((transformed,values),axis=1)
            transformed = pd.DataFrame(transformed)
            

            if(self.verbose):
                print("\n final transitions", transformed.shape)
                print(transformed.head(5))

            return transformed

        except Exception as e:
            print(type(e),e)           
            raise(e)
    
    def transform_transitions(self, transitions, n_in=1, n_out=1, dropnan=True):
        try:
            #n_vars = 1 if type(transitions) is list else transitions.shape[1]
            n_vars = len(transitions[0]-1) if type(transitions) is list else transitions.shape[1]
            df = DataFrame(transitions)
            if(self.verbose):
                print("df:", df.shape)
                print(df.head(5))
            cols = list()
            names = list()
            # input sequence
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j+1, i))
                              for j in range(n_vars)]
            # put it all together
            agg = concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
            return agg
        except Exception as e:
            print(type(e),e)            
            raise(e)

    # Called in train_q_prediction: step 3
    def get_train_test_sets(self, data):
        n_train = int(((80 * self.transitions_real_batch_size)/100))
        train = data[0:n_train, :]
        test = data[n_train:, :]        
        train_X = train[:, :-1]
        train_y = train[:, -1]
        test_X = test[:, :-1]
        test_y = test[:, -1]       
        if(self.verbose):
            print("\n (train_X, train_y), (test_X, test_y)")
            print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        # reshape to 3d to lstm
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        if(self.verbose):
            print("\n after reshape train_X and test_X")
            print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        return train_X, train_y, test_X, test_y

    def get_train(self, data):             
        train_X = data[:, :-1]
        train_y = data[:, -1]            
        if(self.verbose):
            print("\n (train_X, train_y)")
            print(train_X.shape, train_y.shape)
        # reshape to 3d to lstm
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        

        if(self.verbose):
            print("\n after reshape train_X and test_X")
            print(train_X.shape, train_y.shape)

        return train_X, train_y    
    
    def train_q_prediction_evaluate(self, test_X, test_y):
        # make a prediction
        yhat = self.lstm.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        # invert scaling for forecast
        inv_yhat = np.concatenate((yhat, test_X[:, 0:]), axis=1)
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = np.concatenate((test_y, test_X[:, 0:]), axis=1)
        inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        # calculate RMSE
        rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
        self.train_val_rmse_history.append(rmse)
        if(self.verbose):
            print('Test RMSE: %.3f' % rmse)
        return rmse
    
    def val_rmse(self,x,y):
        _y = self.lstm.predict(x)
        rmse = mean_squared_error(y, _y)
        self.train_val_rmse_history.append(rmse)
        if(self.verbose):
            print('Test RMSE: %.3f' % rmse)
        return rmse


    
    def train_q_prediction(self, n_epochs=15, b_size=16):
        self.training = True
        self.trainPredictionCount += 1
        
        transition_features = self.__load_transition_featrues_from_memory()           
        
        train_X, train_y, test_X, test_y = self.get_train_test_sets(transition_features.values)
        
        history = self.lstm.fit(train_X, train_y, epochs=n_epochs, batch_size=b_size, validation_data=(
                                test_X, test_y), verbose=self.verbose, shuffle=False)
        
        self.prepare_train_log(history,test_X,test_y)
        self.LogLstmTrainHistoy()
        self.reduce_transition_memory()
    
    def train_q_prediction_withou_validation(self, n_epochs=15, b_size=16):
        self.training = True
        self.trainPredictionCount += 1
        
        transition_features = self.__load_transition_featrues_from_memory()           
        
        train_X, train_y= self.get_train(transition_features.values)
        
        history = self.lstm.fit(train_X, train_y,validation_split=0.2, epochs=n_epochs, batch_size=b_size, verbose=self.verbose, shuffle=False)
        
        self.prepare_train_log_withou_val(history,train_X,train_y)
        self.LogLstmTrainHistoy()
        self.reduce_transition_memory()
    
    def prepare_train_log(self,history,test_X,test_y):
        if(self.logTrain):
            tlh = history.history['loss']
            vlh = history.history['val_loss']
            self.train_loss_history.extend(tlh)
            self.train_val_loss_history.extend(vlh)            
            #self.train_q_prediction_evaluate(test_X,test_y)
            self.val_rmse(test_X,test_y)
    
    def prepare_train_log_withou_val(self,history,test_X,test_y):
        if(self.logTrain):
            tlh = history.history['loss']
            #vlh = history.history['val_loss']
            self.train_loss_history.extend(tlh)
            #self.train_val_loss_history.extend(0)            
            #self.train_q_prediction_evaluate(test_X,test_y)
            self.val_rmse(test_X,test_y)


    def predict(self, input_transitions):
        self.training = False
        try:
            transition_features = self.scaler.transform(input_transitions)            
            transition_features = transition_features.reshape((transition_features.shape[0], 1, transition_features.shape[1]))
            predict_result = self.lstm.predict(transition_features)
            return predict_result
        except ValueError as vr:
            raise(vr)

    # Main function to reduce TM to TM'. Called in train_q_prediction after training
    def reduce_transition_memory(self):
        try:
            grouped_transitions = self.transitions_memory.load_transitions_batch_as_features_array_grouped(bsize=self.transitions_default_batch_size, include_done_singnal=True, delete_from_memory=True)
            for key in grouped_transitions:
                t = np.array(grouped_transitions[key])
                rt = np.array(t[0])
                t = t[:,:-2]
                predict_q = self.predict(t)
                q = float(np.max(predict_q))
                rt[ft.T_IDX_Q]=q

                self.transitions_memory_l.update_transition(key,rt)

        except Exception as e:
            print(type(e),e)
            raise(e)
   
    def LogLstmTrainHistoy(self):
        if(self.logTrain and self.logDir != ""):
            for i in range(0, len(self.train_loss_history)):
                lstm_logger.logkv("TrainCount", self.trainPredictionCount)
                lstm_logger.logkv('MeanTrainLoss', self.train_loss_history[i])
                if(len(self.train_val_loss_history)>0):
                    lstm_logger.logkv('MeanValidationLoss', self.train_val_loss_history[i])

                if(i>=(len(self.train_loss_history)-1)):
                    lstm_logger.logkv("FinalMeanRMSE",self.train_val_rmse_history[0])
                else:
                    lstm_logger.logkv("FinalMeanRMSE",-1)                
                
                lstm_logger.dumpkvs()
            
        self.train_loss_history.clear()
        if(len(self.train_val_loss_history)>0):
            self.train_val_loss_history.clear()
        self.train_val_rmse_history.clear()
