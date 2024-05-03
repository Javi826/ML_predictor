#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:24:05 2024
@author: javi
"""
from main.modules.mod_pipeline import mod_pipeline



def mod_process_data(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, fets, n_features, type_split):
    
    
    if type_split == 'TRVAL':
                
        X_train_techi = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid,start_tests, endin_tests, lags, fets, n_features,'X_train_techi')
        X_train_month = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid,start_tests, endin_tests, lags, fets, n_features,'X_train_month')
        #X_train_dweek = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid,start_tests, endin_tests, lags, n_features,'X_train_dweek')
        
        X_valid_techi = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid,start_tests, endin_tests, lags, fets, n_features,'X_valid_techi')
        X_valid_month = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid,start_tests, endin_tests, lags, fets, n_features,'X_valid_month')
        #X_valid_dweek = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid,start_tests, endin_tests, lags, n_features,'X_valid_dweek')
        
        X_train = [X_train_techi, X_train_month]
        X_valid = [X_valid_techi, X_valid_month]
        
        y_valid = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, fets, n_features, 'y_valid')
        y_train = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, fets, n_features, 'y_train')
        
        return X_train, X_valid, y_train, y_valid,
    
    elif type_split == 'TESTS':
            
        X_tests_techi = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, fets, n_features,'X_tests_techi')
        X_tests_month = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, fets, n_features,'X_tests_month')
       # X_tests_dweek = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, n_features,'X_tests_dweek')
        
        X_tests = [X_tests_techi, X_tests_month]
        
        y_tests = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, fets, n_features, 'y_tests')
        
        return X_tests, y_tests
        
    elif type_split == 'DATES':
            
        y_tests_date = mod_pipeline(df_preprocess, start_train, endin_train, start_valid, endin_valid, start_tests, endin_tests, lags, fets, n_features, 'y_tests_date')
    
    return y_tests_date