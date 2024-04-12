#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""

import os
import pandas as pd
import numpy as np
from paths.paths import path_base,folder_preprocessing
from functions.def_functions import filter_data_by_date_range, df_plots, diff_series

def mod_preprocessing (df_build,prepro_start_date,prepro_endin_date,lags):
    print(f'START MODUL mod_preprocessi')
    
    df_build_filter  = filter_data_by_date_range(df_build, prepro_start_date, prepro_endin_date)
    selected_columns =['date','close','returns','direction','momentun','volatility','MA','day_week']
    df_preprocessing = pd.DataFrame(df_build_filter, columns=selected_columns)
    
    df_preprocessing['close']      = diff_series(df_preprocessing['close'], diff=30)
    df_preprocessing['returns']    = np.log(df_preprocessing['close'] / df_preprocessing['close'].shift(1))  
    df_preprocessing['direction']  = np.where(df_preprocessing['returns']>0, 1, 0) 
    df_preprocessing['momentun']   = df_preprocessing['returns'].rolling(5).mean().shift(1)
    df_preprocessing['volatility'] = df_preprocessing['returns'].rolling(20).std().shift(1)
    df_preprocessing['MA']         = df_preprocessing['close'].rolling(200).mean().shift(1)
    
    lags = lags
    cols = []
    for lag in range(1,lags+1):
        col = f'lag_{str(lag).zfill(2)}'
        df_preprocessing[col] = df_preprocessing['returns'].shift(lag)
        cols.append(col)
    df_preprocessing.dropna(inplace=True)
    
    #print(df_preprocessing)
    
    #
    df_preprocessing['date'] = pd.to_datetime(df_preprocessing['date'])
    df_plots(df_preprocessing['date'],df_preprocessing['close'],'date','close','lines')
    
    # SAVE Dataframe
    file_suffix = f"_{str(lags).zfill(2)}_{prepro_start_date}_{prepro_endin_date}.xlsx"
    excel_file_path = os.path.join(path_base, folder_preprocessing, f"df_preprocessing{file_suffix}")
    df_preprocessing.to_excel(excel_file_path, index=False)
    
    print(f'ENDIN MODUL mod_preprocessi')
    print('\n')
    return df_preprocessing