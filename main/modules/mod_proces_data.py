#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:24:05 2024
@author: javi
"""
from modules.mod_pipeline import mod_pipeline



def process_data(df_preprocess, endin_data_train, endin_data_valid, initn_data_valid, lags, n_features):
    
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
    
    return X_train, X_valid, y_train, y_valid