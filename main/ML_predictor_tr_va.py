#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:33:28 2024
@author: javi
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import psutil

import yfinance as yf

from modules.mod_init import *
from paths.paths import file_df_data,folder_csv,path_file_csv,results_path,path_tra_val_results,file_tra_val_results, path_base,folder_tra_val_results
from columns.columns import columns_csv_yahoo,columns_clean_order
from functions.def_functions import set_seeds, class_weight,plots_histograms,plot_loss, plot_accu,plot_aucr, evaluate_history,create_results_df
from modules.mod_dtset_clean import mod_dtset_clean
from modules.mod_preprocessing import mod_preprocessing
from modules.mod_pipeline import mod_pipeline

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard
from keras.layers import LSTM, Dense, Dropout, Input, Embedding, Reshape, concatenate, BatchNormalization
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,roc_curve, roc_auc_score

start_time = time.time()

# YAHOO CALL + SAVE + READING file
#------------------------------------------------------------------------------
symbol = "^GSPC"
start_date = "1980-01-01"
endin_date = "2023-12-31"
sp500_data = yf.download(symbol, start=start_date, end=endin_date)
sp500_data.to_csv(path_file_csv)
df_data = pd.read_csv(path_file_csv, header=None, skiprows=1, names=columns_csv_yahoo)

#CALL module Datacleaning
#------------------------------------------------------------------------------
df_clean = mod_dtset_clean(df_data,start_date,endin_date)

#CALL PREPROCESSING
#------------------------------------------------------------------------------
prepro_start_date = '2000-01-01'
prepro_endin_date = '2019-12-31'
lags = 5 

df_preprocessing = mod_preprocessing(df_clean,prepro_start_date,prepro_endin_date,lags)

# X_train - y_train | X_valid - y_valid SPLIT DATA - CALL PIPELINE
#------------------------------------------------------------------------------
n_features = 1
endin_data_train  = initn_data_valid  = ['2018-01-01']
endin_data_valid  = '2018-12-31'
    
print(f"Starts Processing for lags = {lags} and initn_data_valid = {initn_data_valid}")
print('\n')

X_train_techi = mod_pipeline(df_preprocessing, endin_data_train, endin_data_valid,lags, n_features, 'X_train_techi')
X_train_dweek = mod_pipeline(df_preprocessing, endin_data_train, endin_data_valid,lags, n_features, 'X_train_dweek')
X_valid_techi = mod_pipeline(df_preprocessing, initn_data_valid, endin_data_valid,lags, n_features, 'X_valid_techi')
X_valid_dweek = mod_pipeline(df_preprocessing, initn_data_valid, endin_data_valid,lags, n_features, 'X_valid_dweek')

X_train = [X_train_techi, X_train_dweek]
X_valid = [X_valid_techi, X_valid_dweek]

y_valid = mod_pipeline(df_preprocessing, initn_data_valid, endin_data_valid,lags, n_features, 'y_valid')
y_train = mod_pipeline(df_preprocessing, initn_data_valid, endin_data_valid,lags, n_features, 'y_train')

#INPUTS LAYERS
#------------------------------------------------------------------------------
input_lags = Input(shape=(lags, n_features),name='input_lags')
input_days = Input(shape=(1,),name='input_days')

#VARIABLES
#------------------------------------------------------------------------------
dropout     = 0.1
n_neurons_1 = 20
n_neurons_2 = 10
batch_s     = 32
le_rate     = 0.001
patiences   = 20
optimizers  = 'adam'
#LSTM LAYERS
#------------------------------------------------------------------------------
#lstm 1
lstm_layer1 = LSTM(units=n_neurons_1, dropout=dropout, name='LSTM1', return_sequences=True)(input_lags)
#lstm 2
lstm_layer2 = LSTM(units=n_neurons_2, dropout=dropout, name='LSTM2')(lstm_layer1)

#EMBEDDINGS LAYER
#------------------------------------------------------------------------------
day_week_embedding = Embedding(input_dim=5, output_dim=5)(input_days)
day_week_embedding = Reshape(target_shape=(5,))(day_week_embedding)

#CONCATENATE MODEL + BATCHNORMALIZATION
#------------------------------------------------------------------------------
concatenated     = concatenate([lstm_layer2, day_week_embedding])
batch_normalized = BatchNormalization()(concatenated)

#DENSE LAYER
#------------------------------------------------------------------------------
denses_layer = Dense(10, activation='relu')(batch_normalized)
output_layer = Dense(1,  activation='sigmoid', name='output')(denses_layer)

#MODEL DEFINITION + OPTIMIZER + COMPILE
#------------------------------------------------------------------------------

model     = Model(inputs=[input_lags,input_days], outputs=output_layer)
optimizer = Adam(learning_rate=le_rate)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'AUC'])

#model.summary()

#TRAIN MODEL
#------------------------------------------------------------------------------

set_seeds()
file_model_name = f'version01.keras'
path_keras = (results_path / file_model_name).as_posix()

checkpointer = ModelCheckpoint(filepath=path_keras, verbose=0, monitor='val_accuracy',mode='max',save_best_only=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=patiences, verbose=1, restore_best_weights=True)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir)

history = model.fit(X_train, y_train, 
                    epochs=50, 
                    verbose=0,
                    batch_size=batch_s,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpointer, early_stopping, tensorboard])

#EVALUATE MODEL + SAVE ON DATAFRAME + PRINTS
#------------------------------------------------------------------------------
evaluation_results = evaluate_history(history)
print("Best val_accuracy:", evaluation_results['best_valid_accur'])
print("Last val_accuracy:", evaluation_results['valid_accu'])

df_results = create_results_df(lags, initn_data_valid, dropout, n_neurons_1, batch_s, le_rate, optimizers, patiences, evaluation_results)

#PLOTS TRAIN
#------------------------------------------------------------------------------
plot_loss(history)
plot_accu(history)
plot_aucr(history)

#FILES SAVING
#------------------------------------------------------------------------------

print(f"Ending Processing ending for lags = {lags} and initn_data_valid = {initn_data_valid}")
print('\n')

df_tra_val_results = pd.DataFrame(df_results)
excel_file_path    = os.path.join(path_base, folder_tra_val_results, f"df_tra_val_all.xlsx")
df_tra_val_results.to_excel(excel_file_path, index=False)
print("All Training results saved in: 'tra_val_results/df_tra_val_results.xlsx'")

elapsed_time = time.time() - start_time
hours, minutes = divmod(elapsed_time, 3600)
minutes = minutes / 60  

os.system("afplay /System/Library/Sounds/Ping.aiff")
print(f"Total time taken for the process: {int(hours)} hours, {int(minutes)} minutes")

