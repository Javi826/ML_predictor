#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:33:28 2024
@author: javi
"""

#GOOGLE COLAB
#------------------------------------------------------------------------------
import sys
ruta_directorio_clonado = '/content/ML_predictor'
sys.path.append(ruta_directorio_clonado)

import os
import time
import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')

from main.paths.paths import path_file_csv,path_base,folder_tra_val_results,folder_tests_results
from main.functions.def_functions import plots_loss, plots_accu,plots_aucr, evaluate_history, print_results,time_intervals,cross_training,tests_results
from main.modules.mod_data_build import mod_data_build
from main.modules.mod_preprocess import mod_preprocess
from main.modules.mod_models import build_model, train_model
from main.modules.mod_proces_data import mod_process_data

#from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import accuracy_score

start_time = time.time()
#VISUALIZATION PRINTS
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

#CALL DATACLEANING
#------------------------------------------------------------------------------
start_date = "1980-01-01"
endin_date = "2023-12-31"
df_data    = pd.read_csv(path_file_csv, header=None, skiprows=1, names=['date','open','high','low','close','adj_close','volume'])
df_build   = mod_data_build(df_data,start_date,endin_date)

#CALL PREPROCESSING
#------------------------------------------------------------------------------
prepro_start_date = "2000-01-01"
prepro_endin_date = "2023-12-31"
lags = 20
fets = 1
df_preprocess = mod_preprocess(df_build, prepro_start_date, prepro_endin_date,lags)

#CROSS-VALIDATION X_train - y_train | X_valid - y_valid 
#------------------------------------------------------------------------------
n_features    = 1 
n_years_train = 19
m_years_valid = 1
endin_trains  = ['2019-12-31']
start_tests   = ['2023-01-01']
endin_tests   = ['2023-12-31']
train_interval = time_intervals(df_preprocess, n_years_train, m_years_valid, endin_trains)

#VARIABLES
#------------------------------------------------------------------------------
dropout_ra = [0.1, 0.9]
n_neur1_ra = [40]
le_rate_ra = [0.01, 0.0001]
batchsz_ra = [16,32]
l2_regu_ra = [0.01, 0.0001]
n_neurd_ra = [5,10]


dropout_ra = [0.1]
n_neur1_ra = [40]
batchsz_ra = [32]
le_rate_ra = [0.0001]
l2_regu_ra = [0.0001]
n_neurd_ra = [5]

optimizers = 'Adam'
epochss    = 50
patient    = 25

loops_train_results = []
loops_tests_results = []

for dropout in dropout_ra:
    for n_neur1 in n_neur1_ra:
        for batchsz in batchsz_ra:
            for le_rate in le_rate_ra:
                for l2_regu in l2_regu_ra:
                    for n_neurd in n_neurd_ra:
                        print(f"\nTraining with dropout = {dropout} and le_rate = {le_rate} and n_neur1 = {n_neur1} and batchsz = {batchsz} and l2 = {l2_regu} and n_neurd = {n_neurd} and optimizer = {optimizers}")
                        print("*" * 135)
                        
                        n_neur2_ra = int(n_neur1 / 2)
                        means_train_results = []
                        accuracy_results    = []
                        
                        for interval in train_interval:
                            
                            start_train, endin_train, start_valid, endin_valid = interval
                            start_train, endin_train, start_valid, endin_valid = [[start_train], [endin_train], [start_valid], [endin_valid]]
                            print("-" * 135)
                            print(f"Starts Training for  : {n_years_train} years for training and {m_years_valid} years for validation. \nTrain/Val Interval   : {start_train[0]} to {endin_train[0]} and {start_valid[0]} to {endin_valid[0]}\n")
                              
                            X_train, X_valid, y_train, y_valid  = mod_process_data(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, fets, n_features, 'TRVAL')
                            X_tests, y_tests                    = mod_process_data(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, fets, n_features, 'TESTS')
                            y_tests_date                        = mod_process_data(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, fets, n_features, 'DATES')
                            
                            #BUILD & TRAIN MODEL
                            #------------------------------------------------------------------------------
                            model   = build_model(dropout, n_neur1, n_neur2_ra, n_neurd, le_rate, l2_regu, optimizers, lags, fets, n_features)
                            history = train_model(model, X_train, y_train, X_valid, y_valid, dropout, batchsz, epochss, patient)
                            
                            #EVALUATE MODEL
                            #------------------------------------------------------------------------------
                            dc_results = evaluate_history(lags, n_years_train, m_years_valid, start_train, start_valid, dropout,n_neur1, n_neurd, batchsz,le_rate,l2_regu, optimizers,patient,history)
                            means_train_results.append(dc_results)
                            print_results(dc_results)
                            
                            #PLOTS MODEL
                            #------------------------------------------------------------------------------
                            #plots_loss(history)
                            #plots_accu(history)
                            #plots_aucr(history)
                            
                            #MODEL PREDICTIONS
                            #------------------------------------------------------------------------------
                            y_pred     = model.predict(X_tests)
                            y_pred_bin = (y_pred > 0.5).astype(int)
                            
                            tests_accuracy         = accuracy_score(y_tests, y_pred_bin)
                            dc_tests_results       = tests_results(lags, n_years_train, m_years_valid, start_tests, endin_tests, dropout,n_neur1,n_neurd, batchsz, le_rate, l2_regu, optimizers,patient,tests_accuracy)
                            loops_tests_results.append(dc_tests_results)  
                            df_loops_tests_results = pd.DataFrame(loops_tests_results) 
                            excel_file_path = os.path.join(path_base, folder_tests_results, f"df_tests_results_{start_train[0]}_{str(n_years_train).zfill(2)}_{str(m_years_valid).zfill(2)}.xlsx")
                            df_loops_tests_results.to_excel(excel_file_path, index=False)                                
                            print('Tests_accuracy:')
                            print(tests_accuracy)                            
                        
                        #CROSS-VALIDATION
                        #------------------------------------------------------------------------------
                        dc_train_results       = cross_training(lags, n_years_train, m_years_valid, start_train, start_valid, dropout,n_neur1,n_neurd, batchsz,le_rate,l2_regu, optimizers,patient,means_train_results)
                        loops_train_results.append(dc_train_results)   
                        df_loops_train_results = pd.DataFrame(loops_train_results)
                        excel_file_path        = os.path.join(path_base, folder_tra_val_results, f"df_train_results_{start_train[0]}_{str(n_years_train).zfill(2)}_{str(m_years_valid).zfill(2)}.xlsx")
                        df_loops_train_results.to_excel(excel_file_path, index=False)
            

print("-" * 135)
print(f"Ending Training for  : {n_years_train} years for training and {m_years_valid} years for validation. \nTrain/Val Interval   : {start_train[0]} to {endin_train[0]} and {start_valid[0]} to {endin_valid[0]}\n")
os.system("afplay /System/Library/Sounds/Ping.aiff")
elapsed_time   = time.time() - start_time
elapsed_hours, elapsed_minutes = divmod(elapsed_time / 60, 60)
print(f"Total time take to train: {int(elapsed_hours)} hours, {int(elapsed_minutes)} minutes")

        