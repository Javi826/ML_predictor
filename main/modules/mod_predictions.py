#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:05:07 2024
@author: javi
"""

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from modules.mod_backtesting import mod_backtesting


def tests_predictions(model, df_preprocess, X_tests, y_tests, start_tests, endin_tests):
    
    start_tests_i = start_tests[0]
    endin_tests_i = endin_tests[0]
    
    df_date_lag_dir = df_preprocess.copy()   
    tests_data      = df_date_lag_dir[(df_date_lag_dir['date'] > start_tests_i) & (df_date_lag_dir['date'] <= endin_tests_i)]

    y_pred     = model.predict(X_tests)
    y_pred_bin = (y_pred > 0.5).astype(int)

    # Prepare data for DataFrame
    y_tests      = pd.Series(y_tests)
    y_pred_bin   = np.squeeze(y_pred_bin)
    y_pred_bin   = pd.Series(y_pred_bin)
    
    #mod_backtesting(tests_data,y_tests,y_pred_bin,start_tests_i,endin_tests_i)   

    # CTest accuracy
    tests_accuracy = accuracy_score(y_tests, y_pred_bin) 
    
    return tests_accuracy
                             