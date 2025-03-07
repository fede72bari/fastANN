# ---------------------------------------------------------------------------
#                              Libraries Import
# ---------------------------------------------------------------------------


# Multiprocessing
import multiprocessing

# Files Management
import gzip
import joblib
import glob
import csv
import json
import os

# Stocks Indicators
# import talib

# Time Management
import datetime
from datetime import datetime, timedelta, date
import time
import pytz
from pytz import timezone


# Math and Sci
import numpy as np
import math
from scipy.signal import argrelextrema
import random
from scipy.signal import find_peaks
from scipy.signal import argrelmax, argrelmin
from sklearn.preprocessing import StandardScaler
from scipy.integrate import simps
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.integrate import simps
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean


# Reporting
import plotly
from plotly.figure_factory import create_candlestick
from plotly.subplots import make_subplots
import plotly.subplots as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import plot
from matplotlib.pylab import rcParams
from xgboost import plot_tree
import seaborn as sns
from tabulate import tabulate
from IPython.display import HTML

# Data Management
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


# Machine Learning
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tensorflow
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, LambdaCallback
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.losses import mean_squared_error


# Optimization
from deap import base, creator, tools, algorithms

# Models Explainablity
#import shap
#import torch

# Binary Classification Specific Metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_score

# General Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Data sources
import yfinance as yf

# Financial indicators
# import talib



class fastANN:

    def __init__(self,
                 X_data = None,
                 Y_data = None,
                 X_train_s = None,
                 Y_train = None,
                 X_test_s = None,
                 Y_test = None,
                 split_type = 'sequential', # 'random' 'sequential'
                 model_relative_width = [1],
                 model_dropout = [0],
                 learning_rate = 0.0003,
                 activation = 'relu',
                 last_layer_activation = 'sigmoid',
                 loss = 'binary_crossentropy',
                 metrics = ['accuracy'],
                 early_stop_monitor_metric = 'val_accuracy',
                 checkpoint_monitor_metric = 'val_accuracy',
                 checkpoint_mode = 'max',
                 early_stop_mode = 'max',
                 history_metrics = ['accuracy', 'val_accuracy'],
                 save_best_only = True,
                 early_stop_patience = 200,
                 autoencoder_mode = False,
                 train_size_rate = 0.7,
                 save_X_Y_data = True,
                 data_storage_path="\\cyPredict\\",
                 model_name = 'ANN',
                 scale_targets=False):
        
        self.model = Sequential()
        self.save_best_only = save_best_only
        
        self.data_storage_path = data_storage_path
        self.model_name = model_name
        
        self.model_relative_width = model_relative_width
        self.model_dropout = model_dropout
        self.learning_rate = learning_rate
        self.activation = activation
        self.last_layer_activation = last_layer_activation
        self.loss = loss
        self.metrics = metrics
        self.early_stop_monitor_metric = early_stop_monitor_metric
        self.checkpoint_monitor_metric = checkpoint_monitor_metric
        self.checkpoint_mode = checkpoint_mode
        self.early_stop_mode = early_stop_mode
        self.early_stop_patience = early_stop_patience
        self.train_size_rate = train_size_rate
        self.autoencoder_mode = autoencoder_mode
        self.history_metrics = history_metrics
        
        self.hyperparameters_file_name = None
        
        current_datetime = datetime.now()
        self.model_training_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
        
        self.save_X_Y_data = save_X_Y_data
        
        self.model_file_name = None
        self.scaler_file_name = None        
        self.training_history_file_name = None
        self.X_data_df_file_name = None
        self.Y_data_df_file_name = None
        
        self.batch_size = 0
        
        self.loss_df = pd.DataFrame()
        
        
        self.model_summary = {}
        
        self.hyperparmeters = {}
        
        
        self.X_data = X_data
        self.Y_data = Y_data
        
        if(X_train_s is None):
            self.X_train_s = pd.DataFrame()
        else:
            self.X_train_s = X_train_s
            
        if(Y_train is None):
            self.Y_train = pd.DataFrame()
        else:
            self.Y_train = Y_train           
            
        if(X_test_s is None):
            self.X_test_s = pd.DataFrame()
        else:
            self.X_test_s = X_test_s               

        if(Y_test is None):
            self.Y_test = pd.DataFrame()
        else:
            self.Y_test = Y_test  

        self.split_type = split_type
        
        self.scaler = StandardScaler()

        self.scale_targets = scale_targets
        self.Y_scaler = StandardScaler() if scale_targets else None
             
        
        self.early_stop_patience_set(self.early_stop_patience)
        
        if((self.X_data is not None) & (self.Y_data is not None)):
            self.split_and_scale(scaler_fit = True)
            
        self.hyperparameters = {}
            
        self.init_hyperparameters()
        
#         self.checkpoint_callback(self.save_best_only)  
            
            
    def init_hyperparameters(self, 
                             model_training_datetime = None,
                             model_file_name = None,
                             scaler_file_name = None,
                             training_history_file_name = None,
                             X_data_df_file_name = None,
                             Y_data_df_file_name = None
                            ):
        
        self.hyperparameters = {
                               'model_training_datetime': model_training_datetime, 
                               'model_name': self.model_name,
                               'save_best_only': self.save_best_only,
                               'model_relative_width': self.model_relative_width,
                               'model_dropout': self.model_dropout,
                               'learning_rate': self.learning_rate, 
                               'activation': self.activation,
                               'last_layer_activation': self.last_layer_activation,
                               'loss': self.loss,
                               'metrics': self.metrics,
                               'early_stop_monitor_metric': self.early_stop_monitor_metric,
                               'checkpoint_monitor_metric': self.checkpoint_monitor_metric,
                               'checkpoint_mode': self.checkpoint_mode,
                               'early_stop_mode': self.early_stop_mode,
                               'history_metrics': self.history_metrics,
                               'save_best_only': self.save_best_only,
                               'early_stop_patience': self.early_stop_patience,
                               'autoencoder_mode': self.autoencoder_mode,
                               'train_size_rate': self.train_size_rate,
                               'batch_size': self.batch_size,
                               'X_feature_names': self.X_data.columns.tolist(),
                               'Y_feature_names': self.Y_data.columns.tolist(),
                               'scaler_type': 'StandardScaler',
                               'split_type': self.split_type, 
                               'data_storage_path': self.data_storage_path,
                               'model_file_name': model_file_name,
                               'scaler_file_name': scaler_file_name,
                               'training_history_file_name': training_history_file_name,
                               'X_data_df_file_name': X_data_df_file_name,
                               'Y_data_df_file_name': Y_data_df_file_name,
                               'scale_targets': self.scale_targets
                              }
        
        
    def set_hyperparameters(self):
        
        self.model_training_datetime = self.hyperparameters['model_training_datetime']
        self.model_name = self.hyperparameters['model_name']
        self.save_best_only = self.hyperparameters['save_best_only']
        self.model_relative_width = self.hyperparameters['model_relative_width']
        self.model_dropout = self.hyperparameters['model_dropout']
        
        self.learning_rate = self.hyperparameters['learning_rate']
        self.activation = self.hyperparameters['activation']
        self.last_layer_activation = self.hyperparameters['last_layer_activation']
        self.loss = self.hyperparameters['loss']
        self.save_best_only = self.hyperparameters['save_best_only']
        
        if('metrics' in self.hyperparameters):
            self.metrics = self.hyperparameters['metrics']
            
        if('early_stop_monitor_metric' in self.hyperparameters):
            self.early_stop_monitor_metric = self.hyperparameters['early_stop_monitor_metric']
            
        if('checkpoint_monitor_metric' in self.hyperparameters):
            self.checkpoint_monitor_metric = self.hyperparameters['checkpoint_monitor_metric']
        
        if('checkpoint_mode' in self.hyperparameters):
            self.checkpoint_mode = self.hyperparameters['checkpoint_mode']
            
        if('early_stop_mode' in self.hyperparameters):
            self.early_stop_mode = self.hyperparameters['early_stop_mode']
        
        if('history_metrics' in self.hyperparameters):
            self.history_metrics = self.hyperparameters['history_metrics']
        
        if('autoencoder_mode' in self.hyperparameters):
            self.autoencoder_mode = self.hyperparameters['autoencoder_mode']
        
        self.early_stop_patience = self.hyperparameters['early_stop_patience']
        self.train_size_rate = self.hyperparameters['train_size_rate']
        self.batch_size = self.hyperparameters['batch_size']
        self.X_feature_names = self.hyperparameters['X_feature_names']
        self.Y_feature_names = self.hyperparameters['Y_feature_names']
        
        self.scaler_type = self.hyperparameters['scaler_type']
        self.split_type = self.hyperparameters['split_type']
        if(self.hyperparameters['data_storage_path'] is not None):
            self.data_storage_path = self.hyperparameters['data_storage_path']
        self.model_file_name = self.hyperparameters['model_file_name']
        self.scaler_file_name = self.hyperparameters['scaler_file_name']
        
        self.training_history_file_name = self.hyperparameters['training_history_file_name']
        self.X_data_df_file_name = self.hyperparameters['X_data_df_file_name']
        self.Y_data_df_file_name = self.hyperparameters['Y_data_df_file_name']

        self.scale_targets = self.hyperparameters.get('scale_targets', False)

#         if self.scale_targets:
#             self.Y_scaler = joblib.load(self.data_storage_path + "Y_scaler.pkl")
        
                

    def early_stop_patience_set(self, patience):
        
        self.early_stop = EarlyStopping(monitor = self.early_stop_monitor_metric, mode = self.early_stop_mode, verbose = 1, patience = patience)
            
        
    def checkpoint_callback(self, save_best_only):
        
#         model_file_name = self.model_training_datetime + ' - ANN MODEL - ' + self.model_name + '.keras'
        print(f"self.hyperparameters['data_storage_path']  {self.hyperparameters['data_storage_path']}")

        self.checkpoint_callback = ModelCheckpoint(self.hyperparameters['data_storage_path'] + self.hyperparameters['model_file_name'], 
                                                   monitor= self.checkpoint_monitor_metric, 
                                                   mode= self.checkpoint_mode , 
                                                   verbose=1, 
                                                   save_best_only=save_best_only)
        
        
    def network_structure_set_compile(self):
        
        # reset
        self.model = Sequential()
        
        print(len(self.X_train_s))
        
        # first layer
        self.model.add(Dense(self.X_train_s.shape[1] * self.model_relative_width[0] , 
                             input_shape = (self.X_train_s.shape[1],) #, 
#                              activation = self.activation
                            )
                      )
        
        if(self.activation != 'PReLU'):
            self.model.add(Dense(self.X_train_s.shape[1] * self.model_relative_width[0] , 
                             input_shape = (self.X_train_s.shape[1],), 
#                              activation = self.activation
                            )
                      )

        else:
            self.model.add(Dense(self.X_train_s.shape[1] * self.model_relative_width[0] , 
                             input_shape = (self.X_train_s.shape[1],) #, 
#                              activation = self.activation
                            )
                      )
            self.model.add(PReLU())
            
        self.model.add(Dropout(self.model_dropout[0]))


        # inner layers
        for i in range(1, len(self.model_relative_width) ):# - 1):
            
            model_relative_width = self.model_relative_width[i]
            model_dropout = self.model_dropout[i]
            
#             print(f'model_relative_width layer {i}: {model_relative_width}')
#             print(f'model_dropout layer {i}: {model_dropout}')

            
#             self.model.add(Dense(int(self.X_train_s.shape[1] * model_relative_width))) #, activation = self.activation))
        
            if(self.activation != 'PReLU'):
                self.model.add(Dense(int(self.X_train_s.shape[1] * model_relative_width), activation = self.activation))
            else:
                self.model.add(Dense(int(self.X_train_s.shape[1] * model_relative_width))) #, activation = self.activation))
                self.model.add(PReLU())
                
            self.model.add(Dropout(model_dropout)) 
            
            
            
        if self.autoencoder_mode:
            # last layer: ensure it matches input size in autoencoder mode
            self.model.add(Dense(self.X_train_s.shape[1], activation=self.last_layer_activation))  # Match input size
            
        else:
            # last layer: ensure it matches targets size in not autoencoder mode
            
            self.model.add( Dense( int(self.Y_train.shape[1]), activation = self.last_layer_activation ) )
#             self.model.add(Dense(1, activation=self.last_layer_activation))  # Single neuron output


        # compile
        self.model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.0003),                      
                         loss = self.loss, 
                         metrics = self.metrics) #['accuracy'])
        
            
        # report
        self.model_summary = self.model.summary()
        
        display(self.model_summary)


        
        
    def network_training(self, epochs, batch_size):
        
        self.batch_size = batch_size
        
        current_datetime = datetime.now()
        self.model_training_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
        
        
        model_file_name = self.model_training_datetime + ' - ANN MODEL - ' + self.model_name + '.keras'
        scaler_file_name = self.model_training_datetime + ' - SCALER FOR ANN MODEL - ' + self.model_name +'.pkl'
        y_scaler_file_name = self.model_training_datetime + ' - Y SCALER FOR ANN MODEL - ' + self.model_name +'.pkl'
        training_history_file_name = self.model_training_datetime + ' - TRAINING HISTORY OF ANN MODEL - ' + self.model_name + '.csv'
        hyperparameters_file_name = self.model_training_datetime + ' - HYPERPARAMETERS OF ANN MODEL - ' + self.model_name + ".json"
        self.hyperparameters_file_name = hyperparameters_file_name
        
        if(self.save_X_Y_data):
            X_data_df_file_name = self.model_training_datetime + ' - X_data FOR ANN MODEL - ' + self.model_name + '.csv'
            Y_data_df_file_name = self.model_training_datetime + ' - Y_data FOR ANN MODEL - ' + self.model_name + '.csv'
            
            self.X_data.to_csv(self.data_storage_path + X_data_df_file_name, index=False)
            self.Y_data.to_csv(self.data_storage_path + Y_data_df_file_name, index=False)
            
            print('\nsave_X_Y_data saved.')
            
        else:
            X_data_df_file_name = None
            Y_data_df_file_name = None 
            
            print('\nsave_X_Y_data not saved.')
            
        # save hyperparameters
        print('\nInit hyperparameters.')
        self.init_hyperparameters(
                                  model_training_datetime = self.model_training_datetime,
                                  model_file_name = model_file_name,
                                  scaler_file_name = scaler_file_name,
                                  training_history_file_name = training_history_file_name,
                                  X_data_df_file_name = X_data_df_file_name,
                                  Y_data_df_file_name = Y_data_df_file_name
                                 )
        
        print('\nSave hyperparameters.')
        self.save_hyperparameters(hyperparameters_file_name)
        
        # save used scaler
        joblib.dump(self.scaler, self.data_storage_path + scaler_file_name)

        if(self.scale_targets == True):
            joblib.dump(self.Y_scaler, self.data_storage_path + y_scaler_file_name)

        
        self.checkpoint_callback(self.save_best_only)  
        
       # Adjusted lines for autoencoder mode scaling
        # Use scaled data for both training and validation
        if self.autoencoder_mode:
            # For autoencoders, both input and output are scaled the same way
            X_train = self.X_train_s
            Y_train = self.X_train_s  # **Adjusted Line: Use scaled X_train as target**
            X_val = self.X_test_s
            Y_val = self.X_test_s  # **Adjusted Line: Use scaled X_test as target**
        else:
            # Normal ANN mode: use scaled inputs, and potentially unscaled outputs
            X_train = self.X_train_s
            Y_train = self.Y_train_s if self.scale_targets else self.Y_train
            X_val = self.X_test_s
            Y_val = self.Y_test_s if self.scale_targets else self.Y_test


        # Model training logic
        self.model.fit(
            x=X_train,
            y=Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[self.early_stop, self.checkpoint_callback]
        )
        
        # save history
        self.loss_df = pd.DataFrame(self.model.history.history)
        self.loss_df.to_csv(self.data_storage_path + training_history_file_name)
                
        # keep the best model
        self.model = self.load_model(model_file_name)
        
        # plot history
        self.plot_training_history()
        
        
        
    def save_hyperparameters(self, file_name):
        
        with open(self.data_storage_path + file_name, "w") as file:
            json.dump(self.hyperparameters, file)
            
            
    def load_hyperparameters(self, file_name, file_path_name = None):
        
        if(file_path_name is not None):
            self.data_storage_path = file_path_name
        
        print(f'\nTrying to load hyperparameters {self.data_storage_path + file_name}')
        with open(self.data_storage_path + file_name, "r") as file:            
            self.hyperparameters = json.load(file)
        print(f'Hyperparameters loaded.')
        
        
    def load_model(self, model_file_name, file_path_name = None):
        
        if(file_path_name is not None):
            self.data_storage_path = file_path_name
            
        model_file_path = self.data_storage_path + model_file_name
        if os.path.exists(model_file_path):
            print("Model file exists.")
            file_size = os.path.getsize(model_file_path)
            print(f"File size: {file_size} bytes")
            if file_size == 0:
                print("Warning: The model file is empty.")
        else:
            print("Error: Model file does not exist.")

        print(f'\nTrying to load model {self.data_storage_path + model_file_name}')
        self.model = load_model(self.data_storage_path + model_file_name)
        print(f'Model loaded.')
        
        if self.model:
            print("Model is correctly loaded and accessible.")
            self.model.summary()
        else:
            print("Model is None after loading. Check the loading logic.")
        

        
        
    def model_predict(self, data, apply_scaler=True, descale_result=True):
        if apply_scaler:
            print('Scaler applied.')
            data = self.scaler.transform(data)

        predictions = self.model.predict(data)

        if descale_result and self.scale_targets:
            predictions = self.Y_scaler.inverse_transform(predictions)

        return predictions

            
            
            
    def load_scaler(self, scaler_file_name = None, Y_scaler_file_name = None, file_path_name = None):
        
        if(file_path_name is not None):
            self.data_storage_path = file_path_name            
            
        if(scaler_file_name is None):
            scaler_file_name = self.model_training_datetime + ' - SCALER FOR ANN MODEL - ' + self.model_name +'.pkl'
            
        if(Y_scaler_file_name is None):
            y_scaler_file_name = self.model_training_datetime + ' - Y SCALER FOR ANN MODEL - ' + self.model_name +'.pkl'
            
        print(f"self.data_storage_path {self.data_storage_path}")
        print(f"scaler_file_name {scaler_file_name}")
        print(f"scaler_file_name {scaler_file_name}")
            

        print(f"\nTrying to load scaler {self.data_storage_path}{scaler_file_name}")
        self.scaler = joblib.load(self.data_storage_path + scaler_file_name)
        print(f'Scaler loaded.')
        
        if(self.scale_targets == True):
            print(f"\nTrying to load Y_scaler {self.data_storage_path}{y_scaler_file_name}")
            self.Y_scaler = joblib.load(self.data_storage_path + y_scaler_file_name)
            print(f'Y_scaler loaded.')


            
    def load_training_history(self, training_history_file_name, file_path_name = None):
        
        if(file_path_name is not None):
            self.data_storage_path = file_path_name
        
        print(f"\nTrying to load training history {self.data_storage_path + training_history_file_name}")
        self.loss_df = pd.read_csv(self.data_storage_path + training_history_file_name)
        print(f'Training history loaded.')
            
            
    def load_all(self, hyperparameters_file_name = None, file_path_name = None):
        
        if(file_path_name is not None):
            self.data_storage_path = file_path_name
            
        if(hyperparameters_file_name is not None):
            self.hyperparameters_file_name = hyperparameters_file_name
            
        print(f'\nTrying to open hyperparameters file {hyperparameters_file_name}')
        self.load_hyperparameters(self.hyperparameters_file_name)   
        
        print(f'\nSetting hyperparameters')
        self.set_hyperparameters()
        
        print(f'Hyperparameters:\n')   
        display(self.hyperparameters)
        
        self.load_model(self.hyperparameters['model_file_name'])
        
        print(f"\nTrying to import ANN scalers")
        self.load_scaler()
        
        print(f"\nTrying to import ANN training history {self.hyperparameters['training_history_file_name']}")
        self.load_training_history(self.hyperparameters['training_history_file_name'])
        
        # load X and Y data, split and scale
        if((self.hyperparameters['X_data_df_file_name'] is not None) and (self.hyperparameters['Y_data_df_file_name'] is not None)):
            
            self.X_data = pd.read_csv(self.data_storage_path + self.hyperparameters['X_data_df_file_name'])
            self.Y_data = pd.read_csv(self.data_storage_path + self.hyperparameters['Y_data_df_file_name'])
            
            self.split_and_scale(scaler_fit = False) # scaler is loaded not need to fit it again
            
            
        if self.model:
            print("Model is STILL correctly loaded and accessible.")
            self.model.summary()
        else:
            print("Model is None after loading. Check the loading logic.")
        
        
        
    def network_predictions_evaluation(self, min_probability, output_dict = False):       

        print(f'len X_test_s {len(self.X_test_s)}')
        print(f'len Y_test {len(self.Y_test)}')
        # Cut off predictions with low probability
        predictions = self.model.predict(self.X_test_s)
        predictions_df = pd.DataFrame(predictions)
        filtered_predictions_results_df = pd.DataFrame()
        
        if(self.Y_data is not None):
            Y = self.Y_data
        elif(self.Y_train is not None):
            Y = self.Y_train
            
        
        
        count = 0
        for col_name, col_data in Y.items():           
              
            filtered_predictions_results_df[col_name] = predictions_df[count].apply(lambda x: 1 if x > min_probability else 0 ).values
            report = classification_report(self.Y_test[col_name], filtered_predictions_results_df[col_name], output_dict = output_dict)
                          
            print(report)
            
            count += 1
        
        if(output_dict == False):
            return filtered_predictions_results_df, predictions_df
        
        elif(output_dict == True):
            return filtered_predictions_results_df, predictions_df, report
        
    
    def plot_training_history(self):
            
        self.loss_df[self.history_metrics].plot()
        
        
    def split_and_scale(self, scaler_fit = False):        
        
#         if(self.autoencoder_mode == False):

        print(f'self.split_type {self.split_type}')
            
        if(self.split_type == 'sequential'):
            train_size = int(len(self.X_data) * self.train_size_rate)
            test_size = len(self.X_data) - train_size

            self.X_train = self.X_data.head(train_size)
            self.Y_train = self.Y_data.head(train_size)
            self.X_test = self.X_data.tail(test_size)
            self.Y_test = self.Y_data.tail(test_size)

            print('split_and_scale, SEQUENTIAL split')
            print(f"\tShape of Y_train: {self.Y_train.shape}")
            print(f"\tShape of Y_test: {self.Y_test.shape}")
            print(f"\tShape of X_train: {self.X_train.shape}")
            print(f"\tShape of X_test: {self.X_test.shape}")
            
        if(self.split_type == 'random'):
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                                                                self.X_data, 
                                                                self.Y_data, 
                                                                test_size = self.train_size_rate, 
                                                                random_state = 42
                                                               )
            print('split_and_scale, RANDOM split')
            print(f"\tShape of Y_train: {self.Y_train.shape}")
            print(f"\tShape of Y_test: {self.Y_test.shape}")
            print(f"\tShape of X_train: {self.X_train.shape}")
            print(f"\tShape of X_test: {self.X_test.shape}")

        # scale
        if(scaler_fit == True):
            print('\tFit transform X_train')
            self.X_train_s = self.scaler.fit_transform(self.X_train)
        else:
            print('\tOnly transform X_train')
            self.X_train_s = self.scaler.transform(self.X_train)

        print('\tTransform X_test')
        self.X_test_s = self.scaler.transform(self.X_test)

        if self.scale_targets:
            if scaler_fit:
                self.Y_train_s = self.Y_scaler.fit_transform(self.Y_train)
            else:
                self.Y_train_s = self.Y_scaler.transform(self.Y_train)
        
            self.Y_test_s = self.Y_scaler.transform(self.Y_test)
        else:
            self.Y_train_s = self.Y_train
            self.Y_test_s = self.Y_test
          
        print('split_and_scale, end data length')
        print(f"\tShape of Y_train: {self.Y_train.shape}")
        print(f"\tShape of Y_test: {self.Y_test.shape}")
        print(f"\tShape of X_train: {self.X_train.shape}")
        print(f"\tShape of X_test: {self.X_test.shape}")
        print(f"\tShape of X_test_s: {self.X_test_s.shape}")
        

        

    def binary_precision_recall_vs_scoring(self, n_points = 15, plot = True):
        
        # Inizializza le liste per memorizzare i valori di precision e recall
        precision_list = []
        recall_list = []
        cutoff_values = []

        # Ciclo for per calcolare i risultati per ogni valore di cutoff
        for cutoff in range(n_points, 100, 1):
            cutoff_value = cutoff / 100  # Calcola il valore di cutoff da 0 a 1
            print(f'Evaluting cutoof value = {cutoff_value}')
            
            filtered_predictions_results_df, dictionary = self.network_predictions_evaluation(cutoff_value, output_dict=True)

            # Aggiungi i valori di precision e recall alla lista
            precision_list.append(dictionary['1']['precision'])
            recall_list.append(dictionary['1']['recall'])
            cutoff_values.append(cutoff_value)

        # Crea un DataFrame pandas con i valori di precision, recall e cutoff
        df = pd.DataFrame({'Cutoff': cutoff_values, 'Precision': precision_list, 'Recall': recall_list})

        if(plot == True):

            # Plot di precision e recall in funzione di cutoff utilizzando Plotly
            fig = go.Figure()

            # Aggiungi linea per precision
            fig.add_trace(go.Scatter(x=df['Cutoff'], y=df['Precision'], mode='lines', name='Precision'))

            # Aggiungi linea per recall
            fig.add_trace(go.Scatter(x=df['Cutoff'], y=df['Recall'], mode='lines', name='Recall'))

            # Imposta i titoli degli assi e il titolo del grafico
            fig.update_layout(
                xaxis_title='Cutoff',
                yaxis_title='Value',
                title='Precision and Recall vs Cutoff'
            )

            # Mostra il grafico interattivo
            fig.show()
        
        return df
    

    def compute_gradients(self, inputs, targets):
        with tensorflow.GradientTape() as tape:
            tape.watch(inputs) 
            predictions = self.model(inputs)
            
            mse = tensorflow.keras.losses.MeanSquaredError()
            loss = mse(targets, predictions)
#             loss = mean_squared_error(targets, predictions) # tensorflow.keras.losses.mean_squared_error(targets, predictions)
        return tape.gradient(loss, inputs)
    

    def gradient_feature_importance(self, feature_names):
        # Convert X to a tensorflow.Tensor
        X_tensor = tensorflow.convert_to_tensor(self.X_test_s, dtype=tensorflow.float32)

        # Calcola i gradienti
        gradients = self.compute_gradients(X_tensor, self.Y_test)

        # Calcola l'importanza delle feature
        feature_importance = np.mean(np.abs(gradients), axis=0)

        # Normalizza l'importanza
        feature_importance = feature_importance / np.sum(feature_importance)

        # Ordina le feature per importanza
        sorted_idx = np.argsort(feature_importance)
        sorted_features_names = [feature_names[i] for i in sorted_idx]
        sorted_features_importance = [feature_importance[i] for i in sorted_idx]


        # Grafico con Plotly
        fig = go.Figure(go.Bar(
            x=sorted_features_importance,
            y=sorted_features_names,
            orientation='h'
        ))
        fig.update_layout(
            title='Feature Importance (Gradient-based)',
            xaxis_title='Normalized Importance',
            yaxis_title='Features',
            height=800
        )
        fig.show()

        return sorted_features_importance, sorted_features_names

