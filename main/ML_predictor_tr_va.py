#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:33:28 2024
@author: javi
"""

import os
import time
import pandas as pd
#import numpy as np

from modules.mod_init import *
from paths.paths import path_file_csv,path_base,folder_tra_val_results, path_keras
from functions.def_functions import set_seeds,plot_loss, plot_accu,plot_aucr, evaluate_history,create_results_df
from modules.mod_data_build import mod_data_build
from modules.mod_preprocess import mod_preprocess
from modules.mod_pipeline import mod_pipeline


from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Input, Embedding, Reshape, concatenate, BatchNormalization
from keras.regularizers import l1, l2, l1_l2

start_time = time.time()

#CALL DATACLEANING
#------------------------------------------------------------------------------
start_date = "1980-01-01"
endin_date = "2023-12-31"
df_data    = pd.read_csv(path_file_csv, header=None, skiprows=1, names=['date','open','high','low','close','adj_close','volume'])
df_build   = mod_data_build(df_data,start_date,endin_date)


#CALL PREPROCESSING
#------------------------------------------------------------------------------
prepro_start_date = '2000-01-01'
prepro_endin_date = '2019-12-31'
lags = 5

df_preprocess = mod_preprocess(df_build, prepro_start_date, prepro_endin_date,lags)

# X_train - y_train | X_valid - y_valid SPLIT DATA - CALL PIPELINE
#------------------------------------------------------------------------------
n_features = 1
endin_data_train  = initn_data_valid = ['2018-01-01']
endin_data_valid  = '2018-12-31'
    
print(f"Starts Processing for lags = {lags} and initn_data_valid = {initn_data_valid}\n")

X_train_techi = mod_pipeline(df_preprocess, endin_data_train, endin_data_valid,lags, n_features, 'X_train_techi')
X_train_month = mod_pipeline(df_preprocess, endin_data_train, endin_data_valid,lags, n_features, 'X_train_month')
X_train_dweek = mod_pipeline(df_preprocess, endin_data_train, endin_data_valid,lags, n_features, 'X_train_dweek')

X_valid_techi = mod_pipeline(df_preprocess, initn_data_valid, endin_data_valid,lags, n_features, 'X_valid_techi')
X_valid_month = mod_pipeline(df_preprocess, initn_data_valid, endin_data_valid,lags, n_features, 'X_valid_month')
X_valid_dweek = mod_pipeline(df_preprocess, initn_data_valid, endin_data_valid,lags, n_features, 'X_valid_dweek')

X_train = [X_train_techi, X_train_month, X_train_dweek]
X_valid = [X_valid_techi, X_valid_month, X_valid_dweek]

y_valid = mod_pipeline(df_preprocess, initn_data_valid, endin_data_valid,lags, n_features, 'y_valid')
y_train = mod_pipeline(df_preprocess, initn_data_valid, endin_data_valid,lags, n_features, 'y_train')

#INPUTS LAYERS
#------------------------------------------------------------------------------
input_lags   = Input(shape=(lags, n_features),name='input_Lags')
input_months = Input(shape=(12,),name='input_Months')
input_days   = Input(shape=(1,),name='input_Days')

#VARIABLES
#------------------------------------------------------------------------------
dropout_range = 0.1
n_neur1_range = 50
n_neur2_range = int(n_neur1_range // 2)
batch_s_range = 32
le_rate_range = 0.001
patiens_range = 100
optimizers    = 'adam'

#LSTM LAYERS
#------------------------------------------------------------------------------
lstm_layer1 = LSTM(units=n_neur1_range, dropout=dropout_range, name='LSTM1', return_sequences=True)(input_lags)
lstm_layer2 = LSTM(units=n_neur2_range, dropout=dropout_range, name='LSTM2')(lstm_layer1)

#EMBEDDINGS LAYER
#------------------------------------------------------------------------------
dweek_embedding = Embedding(input_dim=5, output_dim=5)(input_days)
dweek_embedding = Reshape(target_shape=(5,))(dweek_embedding)

#CONCATENATE MODEL + BATCHNORMALIZATION
#------------------------------------------------------------------------------
merge_concatenat = concatenate([lstm_layer2, input_months, dweek_embedding])
batch_normalized = BatchNormalization()(merge_concatenat)

#DENSE LAYER
#------------------------------------------------------------------------------
denses_layer = Dense(10, activation='relu')(batch_normalized)
output_layer = Dense(1,  activation='sigmoid', name='output')(denses_layer)

#MODEL DEFINITION + OPTIMIZER + COMPILE
#------------------------------------------------------------------------------
model     = Model(inputs=[input_lags,input_months, input_days], outputs=output_layer)
optimizer = Adam(learning_rate=le_rate_range)
model.compile(optimizer=optimizers,loss='binary_crossentropy',metrics=['accuracy', 'AUC'])
#model.summary()

#TRAIN MODEL
#------------------------------------------------------------------------------
set_seeds()
check_pointers = ModelCheckpoint(filepath=path_keras, verbose=0, monitor='val_accuracy',mode='max',save_best_only=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=patiens_range, verbose=0, restore_best_weights=True)

history = model.fit(X_train, y_train, 
                    epochs=100, 
                    verbose=0,
                    batch_size=batch_s_range,
                    validation_data=(X_valid, y_valid),
                    #class_weight=class_weights,
                    callbacks=[check_pointers, early_stopping])

#EVALUATE MODEL + SAVE ON DATAFRAME + PRINTS
#------------------------------------------------------------------------------
ev_results = evaluate_history(history)
df_results = create_results_df(lags, initn_data_valid, dropout_range, n_neur1_range, batch_s_range, le_rate_range, optimizers, patiens_range, ev_results)

print("Best epoch    Train accuracy :", ev_results['best_train_epoch_accu'])
print("Best epoch    Valid accuracy :", ev_results['best_valid_epoch_accu'])
print("Best epoch    Train AUC      :", ev_results['best_train_epoch_AUC'])
print("Best epoch    Valid AUC      :", ev_results['best_valid_epoch_AUC'])
print("Best accuracy Train data     :", ev_results['best_train_accu'])
print("Best AUC      Train data     :", ev_results['best_train_AUC'])
print("Best accuracy Valid data     :", ev_results['best_valid_accu'])
print("Best AUC      Valid data     :", ev_results['best_valid_AUC'])

#PLOTS TRAIN
#------------------------------------------------------------------------------
plot_loss(history)
plot_accu(history)
plot_aucr(history)

#FILES SAVING
#------------------------------------------------------------------------------
print(f"\nEnding Processing ending for lags = {lags} and initn_data_valid = {initn_data_valid}\n")

df_tra_val_results = pd.DataFrame(df_results)
excel_file_path    = os.path.join(path_base, folder_tra_val_results,f"df_tra_val_all.xlsx")
df_tra_val_results.to_excel(excel_file_path, index=False)
print("All Training results saved in: 'tra_val_results/df_tra_val_results.xlsx'")

os.system("afplay /System/Library/Sounds/Ping.aiff")
elapsed_time   = time.time() - start_time
elapsed_hours, elapsed_minutes = divmod(elapsed_time / 60, 60)
print(f"Total time taken for the process: {int(elapsed_hours)} hours, {int(elapsed_minutes)} minutes")

