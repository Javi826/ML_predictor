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
from functions.def_functions import plots_loss, plots_accu,plots_aucr, evaluate_history,create_results_df, print_results,time_intervals
from modules.mod_data_build import mod_data_build
from modules.mod_preprocess import mod_preprocess
from modules.mod_models import build_model, train_model
from modules.mod_proces_data import mod_process_data

start_time = time.time()

#CALL DATACLEANING
#------------------------------------------------------------------------------
start_date = "1980-01-01"
endin_date = "2023-12-31"
df_data    = pd.read_csv(path_file_csv, header=None, skiprows=1, names=['date','open','high','low','close','adj_close','volume'])
df_build   = mod_data_build(df_data,start_date,endin_date)

#CALL PREPROCESSING
#------------------------------------------------------------------------------
prepro_start_date = "2000-01-01"
prepro_endin_date = "2019-12-31"
lags = 5

df_preprocess = mod_preprocess(df_build, prepro_start_date, prepro_endin_date,lags)


#CROSS-VALIDATION Split
#------------------------------------------------------------------------------
# X_train - y_train | X_valid - y_valid SPLIT DATA - CALL PIPELINE
#------------------------------------------------------------------------------
n_features     = 1 

n_years_train  = 19
m_years_valid  = 1
time_interval  = time_intervals(df_preprocess, n_years_train, m_years_valid)

print(time_interval)

all_train_results = []

for interval in time_interval:
    
    start_train, endin_train, start_valid, endin_valid = interval
    start_train, endin_train, start_valid, endin_valid = [[start_train], [endin_train], [start_valid], [endin_valid]]
      
    X_train, X_valid, y_train, y_valid = mod_process_data(df_preprocess, start_train, endin_train, start_valid, endin_valid, lags, n_features)

    #VARIABLES
    #------------------------------------------------------------------------------
    optimizers = 'adam'
    dropout_ra = 0.1
    n_neur1_ra = 50
    n_neur2_ra = int(n_neur1_ra / 2)
    n_neur3_ra = 10
    le_rate_ra = 0.001
    l2_regu_ra = 0.001
    
    #BUILD MODEL
    #------------------------------------------------------------------------------
    model = build_model(dropout_ra, n_neur1_ra, n_neur2_ra, n_neur3_ra, le_rate_ra, l2_regu_ra, optimizers, lags, n_features)
    
    #TRAIN MODEL
    #------------------------------------------------------------------------------
    batchs_ra = 32
    epochs_ra = 100
    patien_ra = 100
    
    history   = train_model(model, X_train, y_train, X_valid, y_valid, batchs_ra, epochs_ra, patien_ra, path_keras)
    
    #EVALUATE MODEL Training
    #------------------------------------------------------------------------------
    ev_results = evaluate_history(history)
    all_train_results.append(ev_results)
    print_results(ev_results)
    
    #PLOTS MODEL Training
    #------------------------------------------------------------------------------
    plots_loss(history)
    plots_accu(history)
    plots_aucr(history)
    
    #ENDING +  SAVING
    #------------------------------------------------------------------------------
    df_tra_val_results = create_results_df(lags, start_valid, dropout_ra, n_neur1_ra, batchs_ra, le_rate_ra, optimizers, patien_ra, ev_results)
    excel_file_path    = os.path.join(path_base, folder_tra_val_results,f"df_tra_val_all.xlsx")
    df_tra_val_results.to_excel(excel_file_path, index=False)
    print("All Training results saved in: 'tra_val_results/df_tra_val_results.xlsx'")
  

#CROSS-VALIDATION RESULTS
#------------------------------------------------------------------------------
df_train_results = pd.DataFrame(all_train_results)
mean_results     = df_train_results.mean()
print("Mean results:")
print(mean_results)
    
os.system("afplay /System/Library/Sounds/Ping.aiff")
elapsed_time   = time.time() - start_time
elapsed_hours, elapsed_minutes = divmod(elapsed_time / 60, 60)
print(f"Total time taken for the process: {int(elapsed_hours)} hours, {int(elapsed_minutes)} minutes")