#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""

import os
import numpy as np
import pandas as pd
from main.paths.paths import path_base,folder_preprocess
from main.functions.def_functions import filter_data_by_date_range, df_plots, diff_series

def mod_preprocess (df_build,prepro_start_date,prepro_endin_date,lags, rets, e_features):
    
    #RETURNS
    #---------------------------------------------------------------------------------------------- 
    
    df_build_filter  = filter_data_by_date_range(df_build, prepro_start_date, prepro_endin_date) 

    df_preprocess                   = df_build_filter.copy()  
    returns_diff                    = df_preprocess['close'] - df_preprocess['close'].shift(1)
    df_preprocess['nday_direction'] = np.where(np.sign(returns_diff) > 0, 1, 0)
    #df_preprocess['close']          = diff_series(df_preprocess['close'], diff=30)
    df_preprocess['returns']        = np.log(df_preprocess['close'] / df_preprocess['close'].shift(rets))
    df_preprocess['returns_diff']   = diff_series(df_preprocess['returns'], diff=30)
    df_preprocess['direction']      = np.where(df_preprocess['returns']>0, 1, 0)
    #df_preprocess['direction']      = df_preprocess['direction'].shift(1)
    
    
    lags = lags
    cols = []
    for lag in range(1,lags+1):
        col = f'lag_{str(lag).zfill(2)}'
        df_preprocess[col] = df_preprocess['returns_diff'].shift(lag)
        cols.append(col) 
        
    #FEATURING
    
    #----------------------------------------------------------------------------------------------    
    if e_features == 'Yes':
        
       df_preprocess['fet_momentun']   = diff_series(df_preprocess['returns'].rolling(20).mean(),diff=30)
       df_preprocess['fet_volatility'] = diff_series(df_preprocess['returns'].rolling(20).std(),diff=30)
       df_preprocess['fet_volume']     = diff_series(df_preprocess['volume'],diff=30)
        
       #Generar las columnas de retraso para las características adicionales
       fet_cols = [col for col in df_preprocess.columns if col.startswith('fet_')]
       lagged_features = []
        
       for col in fet_cols:
           lagged_feature_cols = pd.concat([df_preprocess[col].shift(lag).rename(f'lag_fet_{str(lag).zfill(2)}_{col[4:]}') for lag in range(1, lags + 1)], axis=1)
           lagged_features.append(lagged_feature_cols)
        
    # Concatenar las características adicionales retrasadas al dataframe principal
       df_preprocess = pd.concat([df_preprocess] + lagged_features, axis=1)
                
    #DROPNA
    #----------------------------------------------------------------------------------------------  
     
    df_preprocess.dropna(inplace=True)
    
    #df_preprocessing['date'] = pd.to_datetime(df_preprocessing['date'])
    df_plots(df_preprocess['date'],df_preprocess['close'],'date','close','lines')
    
    # SAVE Dataframe
    file_suffix     = f"_{str(lags).zfill(2)}_{prepro_start_date}_{prepro_endin_date}.xlsx"
    excel_file_path = os.path.join(path_base, folder_preprocess, f"df_preprocess{file_suffix}")
    df_preprocess.to_excel(excel_file_path, index=False)
    
    return df_preprocess