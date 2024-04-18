#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""

import os
import pandas as pd
import numpy as np
from paths.paths import path_base,folder_preprocess
from functions.def_functions import filter_data_by_date_range, df_plots, diff_series

def mod_preprocess (df_build,prepro_start_date,prepro_endin_date,lags):
    print(f'\nSTARTS MODUL mod_preprocess')
    
    df_build_filter  = filter_data_by_date_range(df_build, prepro_start_date, prepro_endin_date)    
    
    df_preprocess                 = df_build_filter.copy()  
    df_preprocess['returns']      = np.log(df_preprocess['close'] / df_preprocess['close'].shift(1))  
    df_preprocess['returns_diff'] = diff_series(df_preprocess['returns'], diff=30)
    df_preprocess['direction']    = np.where(df_preprocess['returns']>0, 1, 0) 
    
    lags = lags
    cols = []
    for lag in range(1,lags+1):
        col = f'lag_{str(lag).zfill(2)}'
        df_preprocess[col] = df_preprocess['returns_diff'].shift(lag)
        cols.append(col) 
        
    df_preprocess['momentun']   = df_preprocess['returns'].rolling(5).mean().shift(1)
    df_preprocess['volatility'] = df_preprocess['returns'].rolling(20).std().shift(1)       
    df_preprocess.dropna(inplace=True)
    
    #df_preprocessing['date'] = pd.to_datetime(df_preprocessing['date'])
    df_plots(df_preprocess['date'],df_preprocess['close'],'date','close','lines')
    
    # SAVE Dataframe
    file_suffix = f"_{str(lags).zfill(2)}_{prepro_start_date}_{prepro_endin_date}.xlsx"
    excel_file_path = os.path.join(path_base, folder_preprocess, f"df_preprocess{file_suffix}")
    df_preprocess.to_excel(excel_file_path, index=False)
    
    print(f'ENDING MODUL mod_preprocess \n')
    return df_preprocess