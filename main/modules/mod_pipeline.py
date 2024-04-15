#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler



def mod_pipeline(df_preprocessing, initn_date_range, endin_date_range, lags, n_features, data_type):
    
    
    for cutoff_date in initn_date_range:
        #print(f"Pipeline for {data_type}: start with lags = {lags}, initn_date_range = {cutoff_date}, endin_date_range = {endin_date_range}")
        
        #df_columns = ['date'] + [col for col in df_preprocessing.columns if col.startswith('lag')] + ['direction'] + ['day_week']
        #df_date_lag_dir = df_preprocessing[df_columns].copy()
        df_date_lag_dir = df_preprocessing.copy()
                  
        #DATA SPLIT
        #------------------------------------------------------------------------------  
        
        train_data = df_date_lag_dir[df_date_lag_dir['date'] <= cutoff_date]
        valid_data = df_date_lag_dir[(df_date_lag_dir['date'] > cutoff_date) & (df_date_lag_dir['date'] <= endin_date_range)]
        tests_data = df_date_lag_dir[(df_date_lag_dir['date'] > cutoff_date) & (df_date_lag_dir['date'] <= endin_date_range)]
        
        dlags_columns_selected = [col for col in df_date_lag_dir.columns if col.startswith('lag')]
        month_columns_selected = [col for col in df_date_lag_dir.columns if col.startswith('month')]
               
        #X_TRAIN_techi + dweek
        #------------------------------------------------------------------------------
        
        if data_type == 'X_train_techi':
            
            X_data        = train_data[dlags_columns_selected]
            scaler        = StandardScaler()
            X_scaled      = scaler.fit_transform(X_data)
            X_scaled      = pd.DataFrame(X_scaled, columns=dlags_columns_selected)
            X_reshaped    = X_scaled.values.reshape(-1, lags, n_features)
            X_train_techi = X_reshaped
            
            return X_train_techi
        
        elif data_type == 'X_train_month':
            
            X_train_month  = train_data[month_columns_selected]
            
            return X_train_month
        
        elif data_type == 'X_train_dweek':  
            
            X_train_dweek = train_data['day_week']
            
            return X_train_dweek
            
        #X_VALID
        #------------------------------------------------------------------------------
        
        elif data_type == 'X_valid_techi':
            
            X_data        = valid_data[dlags_columns_selected]
            scaler        = StandardScaler()
            X_scaled      = scaler.fit_transform(X_data)
            X_scaled      = pd.DataFrame(X_scaled, columns=dlags_columns_selected)
            X_reshaped    = X_scaled.values.reshape(-1, lags, n_features)
            X_valid_techi = X_reshaped          

            return X_valid_techi
        
        elif data_type == 'X_valid_month':
            
            X_valid_month  = valid_data[month_columns_selected]
            
            return X_valid_month
        
        elif data_type == 'X_valid_dweek': 
            
            X_valid_dweek = valid_data['day_week']
            
            return X_valid_dweek
        
        
        #X_TESTS
        #------------------------------------------------------------------------------        
        elif data_type == 'X_tests_techi':
            
            X_data        = tests_data[dlags_columns_selected]
            scaler        = StandardScaler()
            X_scaled      = scaler.fit_transform(X_data)
            X_scaled      = pd.DataFrame(X_scaled, columns=dlags_columns_selected)
            X_reshaped    = X_scaled.values.reshape(-1, lags, n_features)
            X_tests_techi = X_reshaped
            
            return X_tests_techi
        
        elif data_type == 'X_tests_month':
            
            X_tests_month  = tests_data[month_columns_selected]
            
            return X_tests_month
        
        elif data_type == 'X_tests_dweek':           
            X_tests_dweek = tests_data['day_week']
            
            return X_tests_dweek
                        

        #y_train,valid,tests
        #------------------------------------------------------------------------------             
        elif data_type == 'y_train':
            
            y_train = train_data['direction']

            return y_train
        
        elif data_type == 'y_valid':
            
            y_valid = valid_data['direction']
            
            return y_valid
        
        elif data_type == 'y_tests':
            
            y_tests = valid_data['direction']
            
            return y_tests
