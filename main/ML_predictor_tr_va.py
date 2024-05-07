
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:33:28 2024
@author: javi
"""
#import sys
import os

#GOOGLE COLAB
#------------------------------------------------------------------------------
#ruta_directorio_clonado = '/content/ML_predictor'
#sys.path.append(ruta_directorio_clonado)

#GOOGLE JUPYTER
#------------------------------------------------------------------------------
#nuevo_directorio = "/home/jupyter/ML_predictor"
#os.chdir(nuevo_directorio)

import time
import warnings
import pandas as pd
import shap
import numpy as np
#import yfinance as yf
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')

start_time = time.time()
#VISUALIZATION PRINTS
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


from main.paths.paths import path_base,folder_csv
from main.functions.def_functions import evaluate_history, print_results,time_intervals #plots_loss, plots_accu,plots_aucr,
from main.modules.mod_models import build_model, train_model
from main.modules.mod_data_build import mod_data_build
from main.modules.mod_preprocess import mod_preprocess
from main.modules.mod_proces_data import mod_process_data
from main.modules.mod_predictions import tests_predictions
from main.modules.mod_save_results import save_train_results, save_tests_results

# YAHOO CALL + SAVE + READING file
#------------------------------------------------------------------------------
symbol     = "^GSPC"
start_date = "1980-01-01"
endin_date = "2023-12-31"
#index_price_data = yf.download(symbol, start=start_date, end=endin_date)

index_file_name = f"{symbol}.csv"
csv_path = os.path.join(path_base, folder_csv, index_file_name)
#index_price_data.to_csv(csv_path, index=True)

#CALL BUILD
#------------------------------------------------------------------------------
build_start_date  = "1980-01-01"
build_endin_date  = "2024-04-30"
df_data           = pd.read_csv(csv_path, header=None, skiprows=1, names=['date','open','high','low','close','adj_close','volume'])
df_build          = mod_data_build(df_data,build_start_date,build_endin_date)


#VARIABLES
#------------------------------------------------------------------------------

#rets_ra    = [10,30]  
#lags_ra    = [10,20]
#n_neur1_ra = [40]
#dropout_ra = [0.1]
#batchsz_ra = [32]
#le_rate_ra = [0.0001]
#l2_regu_ra = [0.0001]
#n_neurd_ra = [5] 

rets_ra      = [1]
lags_ra      = [20]  
dropout_ra   = [0.1] 
n_neur1_ra   = [40]
batchsz_ra   = [32]
le_rate_ra   = [0.0001]
l2_regu_ra   = [0.0001]
n_neurd_ra   = [5]

e_features = 'Yes'

loops_train_results = []
loops_tests_results = []

for rets in rets_ra:
    for lags in lags_ra:
        
        dim_array1 = 1
        dim_arrays = dim_array1 * lags
        n_features = 3
        
        #CALL PREPROCESSING
        prepro_start_date = "2000-01-01"
        prepro_endin_date = "2024-04-30"
        #------------------------------------------------------------------------------
        df_preprocess = mod_preprocess(df_build, prepro_start_date, prepro_endin_date,lags,rets, e_features)
        
        #CROSS-VALIDATION X_train - y_train | X_valid - y_valid 
        #------------------------------------------------------------------------------
        endin_train   = ['2019-12-31']
        start_tests   = ['2023-01-01']
        endin_tests   = ['2023-12-31']
        n_years_train = 19
        m_years_valid = 1
        train_interval = time_intervals(df_preprocess, n_years_train, m_years_valid, endin_train)
        
        optimizers = 'Adam'
        epochss    = 50
        patient    = 25
        
        for dropout in dropout_ra:
            for n_neur1 in n_neur1_ra:
                for batchsz in batchsz_ra:
                    for le_rate in le_rate_ra:
                        for l2_regu in l2_regu_ra:
                            for n_neurd in n_neurd_ra:
                                print(f"\nRets = {rets} | lags= {lags} | dropout= {dropout} | le_rate= {le_rate} | n_neur1= {n_neur1} | batchsz= {batchsz} | l2= {l2_regu} | n_neurd= {n_neurd} | optimizer= {optimizers}")
                                
                                n_neur2_ra = int(n_neur1 / 2)
                                means_train_results = []
                                
                                for interval in train_interval:
                                    
                                    start_train, endin_train, start_valid, endin_valid = interval
                                    start_train, endin_train, start_valid, endin_valid = [[start_train], [endin_train], [start_valid], [endin_valid]]
                                    print("-" * 124)
                                    print(f"Training with  : {n_years_train} years for training and {m_years_valid} years for validation. \nInterval dates : {start_train[0]} to {endin_train[0]} and {start_valid[0]} to {endin_valid[0]}\n")
                                      
                                    X_train, X_valid, y_train, y_valid = mod_process_data(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, dim_arrays, n_features, 'TRVAL')
                                    X_tests, y_tests                   = mod_process_data(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, dim_arrays, n_features, 'TESTS')
                                    y_tests_date                       = mod_process_data(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, dim_arrays, n_features, 'DATES')
                                    
                                    #BUILD & TRAIN MODEL
                                    #------------------------------------------------------------------------------
                                    model   = build_model(dropout, n_neur1, n_neur2_ra, n_neurd, le_rate, l2_regu, optimizers, lags,dim_arrays, n_features)
                                    history = train_model(model, X_train, y_train, X_valid, y_valid, dropout, batchsz, epochss, patient)
                                    #max_length = max(len(x) for x in X_train)
                                    #explainer   = shap.Explainer(model, X_train_np)
                                    #X_train_np = np.array([x[:max_length] + [0]*(max_length-len(x)) for x in X_train])
                                    #shap_values = explainer.shap_values(X_train_np)
                                    
                                    #EVALUATE MODEL
                                    #------------------------------------------------------------------------------
                                    dc_results = evaluate_history(lags, n_years_train, m_years_valid, start_train, start_valid, dropout,n_neur1, n_neurd, batchsz,le_rate,l2_regu, optimizers,patient,history)
                                    means_train_results.append(dc_results)
                                    print_results(dc_results)
                                    
                                    #PLOTS MODEL
                                    #------------------------------------------------------------------------------
                                    #plots_loss(history); plots_accu(history); plots_aucr(history)
                                    
                                    #MODEL PREDICTIONS
                                    #------------------------------------------------------------------------------
                                    tests_accuracy = tests_predictions(model, X_tests, y_tests_date, y_tests) 
                                    print('Tests_accuracy:',tests_accuracy)

                                    #SAVE TESTS RESULTS
                                    #------------------------------------------------------------------------------                           
                                    save_tests_results(rets, lags, n_years_train, m_years_valid, start_train, start_tests, endin_tests, dropout, n_neur1, n_neurd, batchsz, le_rate, l2_regu, optimizers, patient, tests_accuracy, loops_tests_results)
                                    
                                
                                #TRAINING-CROSS-VALIDATION RESULTS
                                #------------------------------------------------------------------------------
                                save_train_results(lags, n_years_train, m_years_valid, start_train, start_valid, dropout,n_neur1,n_neurd, batchsz,le_rate,l2_regu, optimizers,patient,means_train_results,loops_train_results)
            
print("*" * 124)
os.system("afplay /System/Library/Sounds/Ping.aiff")
elapsed_time   = time.time() - start_time
#print(elapsed_time)
elapsed_hours, elapsed_minutes = divmod(elapsed_time / 60, 60)
print(f"Total time take to train: {int(elapsed_hours)} hours, {int(elapsed_minutes)} minutes")

        
