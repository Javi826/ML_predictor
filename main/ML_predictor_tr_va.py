#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:33:28 2024
@author: javi
"""

import os
import time
import pandas as pd

from modules.mod_init import *
from paths.paths import path_file_csv,path_base,folder_tra_val_results, path_keras
from functions.def_functions import plot_loss, plot_accu,plot_aucr, evaluate_history,create_results_df
from modules.mod_data_build import mod_data_build
from modules.mod_preprocess import mod_preprocess
from modules.mod_pipeline import mod_pipeline
from modules.mod_model import build_model,train_model

start_time = time.time()

#CALL DATACLEANING
#------------------------------------------------------------------------------
start_date = "2000-01-01"
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

X_train_techi = mod_pipeline(df_preprocess, endin_data_train, endin_data_valid, lags, n_features, 'X_train_techi')
X_train_month = mod_pipeline(df_preprocess, endin_data_train, endin_data_valid, lags, n_features, 'X_train_month')
X_train_dweek = mod_pipeline(df_preprocess, endin_data_train, endin_data_valid, lags, n_features, 'X_train_dweek')

X_valid_techi = mod_pipeline(df_preprocess, initn_data_valid, endin_data_valid, lags, n_features, 'X_valid_techi')
X_valid_month = mod_pipeline(df_preprocess, initn_data_valid, endin_data_valid, lags, n_features, 'X_valid_month')
X_valid_dweek = mod_pipeline(df_preprocess, initn_data_valid, endin_data_valid, lags, n_features, 'X_valid_dweek')

X_train = [X_train_techi, X_train_month, X_train_dweek]
X_valid = [X_valid_techi, X_valid_month, X_valid_dweek]

y_valid = mod_pipeline(df_preprocess, initn_data_valid, endin_data_valid, lags, n_features, 'y_valid')
y_train = mod_pipeline(df_preprocess, initn_data_valid, endin_data_valid, lags, n_features, 'y_train')

#VARIABLES
#------------------------------------------------------------------------------
n_features = 1
optimizers = 'adam'

dropout_ra = 0.1
n_neur1_ra = 50
n_neur2_ra = int(n_neur1_ra / 2)
n_neur3_ra = 10
batch_s_ra = 32
le_rate_ra = 0.001
l2_regu_ra = 0.001
patiens_ra = 100

#BUILD MODEL
#------------------------------------------------------------------------------
model = build_model(dropout_ra, n_neur1_ra, n_neur2_ra, n_neur3_ra, batch_s_ra, le_rate_ra, l2_regu_ra, optimizers, lags, n_features)

#TRAIN MODEL
#------------------------------------------------------------------------------
patien_ra = 100
epochs_ra = 100
history   = train_model(model, X_train, y_train, X_valid, y_valid, batch_s_ra, epochs_ra, patien_ra, path_keras)

#EVALUATE MODEL + SAVE ON DATAFRAME + PRINTS
#------------------------------------------------------------------------------
ev_results = evaluate_history(history)
df_results = create_results_df(lags, initn_data_valid, dropout_ra, n_neur1_ra, batch_s_ra, le_rate_ra, optimizers, patiens_ra, ev_results)

print("Best epoch    Valid accuracy :", round(ev_results['best_valid_epoch_accu'], 2))
print("Best epoch    Valid AUC      :", round(ev_results['best_valid_epoch_AUC'], 2))
print("Best accuracy Valid data     :", round(ev_results['best_valid_accu'], 2))
print("Best AUC      Valid data     :", round(ev_results['best_valid_AUC'], 2))

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

