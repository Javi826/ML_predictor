#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:05:07 2024
@author: javi
"""
import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from main.paths.paths import path_base,folder_tests_results


def tests_predictions(model, X_tests, y_tests_date, y_tests):
    
    # Model predictions
    y_pred = model.predict(X_tests)
    y_pred_bin = (y_pred > 0.5).astype(int)

    # Prepare data for DataFrame
    y_tests_date   = np.squeeze(y_tests_date)
    y_tests        = np.squeeze(y_tests)
    y_pred_bin     = np.squeeze(y_pred_bin)
    df_predictions = pd.DataFrame({'y_tests_date': y_tests_date, 'y_test': y_tests, 'y_pred_bin': y_pred_bin})

    # Save predictions to Excel
    excel_pred_path = os.path.join(path_base, folder_tests_results, "y_predictions.xlsx")
    df_predictions.to_excel(excel_pred_path, index=False)    
    tests_accuracy  = accuracy_score(y_tests, y_pred_bin)                               

    return tests_accuracy
