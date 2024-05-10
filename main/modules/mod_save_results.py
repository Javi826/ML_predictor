#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:14:17 2024

@author: javi
"""
import os
import pandas as pd
from datetime import datetime
from main.paths.paths import path_base,folder_tra_val_results,folder_tests_results
from main.functions.def_functions import train_results, tests_results

def save_tests_results(rets,lags, n_years_train, m_years_valid, start_train, start_tests, endin_tests, dropout,n_neur1,d_layers, n_neurd, batchsz, le_rate, l2_regu, optimizers, patient,tests_accuracy,loops_tests_results):
    
    start_train_strg = start_train[0]
    start_train_year = start_train_strg[:4]
    
    dc_tests_results       = tests_results(rets, lags, n_years_train, m_years_valid, start_tests, endin_tests, dropout,n_neur1, d_layers, n_neurd, batchsz, le_rate, l2_regu, optimizers,patient,tests_accuracy)
    loops_tests_results.append(dc_tests_results)      
    df_loops_tests_results = pd.DataFrame(loops_tests_results)
    
    file_name        = f"df_tests_results_{start_train_year}_{str(n_years_train).zfill(2)}_{str(m_years_valid).zfill(2)}.xlsx"
    excel_tests_path = os.path.join(path_base, folder_tests_results, file_name)
    df_loops_tests_results.to_excel(excel_tests_path, index=False)
    
    
def save_train_results(lags, n_years_train, m_years_valid, start_train, start_valid, dropout,n_neur1,d_layers, n_neurd, batchsz,le_rate,l2_regu, optimizers,patient,means_train_results,loops_train_results):
    
    start_train_strg = start_train[0]
    start_train_year = start_train_strg[:4]
       
    dc_train_results       = train_results(lags, n_years_train, m_years_valid, start_train, start_valid, dropout,n_neur1,d_layers,n_neurd, batchsz,le_rate,l2_regu, optimizers,patient,means_train_results)
    loops_train_results.append(dc_train_results)     
    df_loops_train_results = pd.DataFrame(loops_train_results)
    
    file_name       = f"df_train_results_{start_train_year}_{str(n_years_train).zfill(2)}_{str(m_years_valid).zfill(2)}.xlsx"
    excel_trva_path = os.path.join(path_base, folder_tra_val_results, file_name)
    df_loops_train_results.to_excel(excel_trva_path, index=False)
