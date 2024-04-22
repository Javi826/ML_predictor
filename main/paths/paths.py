#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 21:06:19 2023
@author: jlahoz
"""

import os
from pathlib import Path

## Directorio para almacenar archivos CSV
path_base = "/Users/javi/Desktop/ML/ML_predictor"

file_df_data           = "sp500_data.csv"
folder_csv             = "inputs/historicyh"
path_file_csv          = os.path.join(path_base, folder_csv, file_df_data)

file_df_build          = "df_build.csv"
folder_df_build        = "inputs/data_build"
path_df_build          = os.path.join(path_base, folder_df_build, file_df_build)

file_summary_stats     = 'df_summary_stats'
folder_summary_stats   = "outputs/summary_stats"
path_summary_stats     = os.path.join(path_base, folder_summary_stats, file_summary_stats)

file_preprocess        = 'df_preprocessing.xlsx'
folder_preprocess      = "inputs/preprocess"
path_preprocess        = os.path.join(path_base, folder_preprocess, file_preprocess)

file_tra_val_results   = 'df_tra_val_results.xlsx'
folder_tra_val_results = "results/tra_val_results"
path_tra_val_results   = os.path.join(path_base, folder_tra_val_results, file_tra_val_results)

file_tra_val_results   = 'df_tra_val_results.xlsx'
folder_mean_results    = "results/mean_results"
path_tra_val_results   = os.path.join(path_base, folder_tra_val_results, file_tra_val_results)

file_tests_results     = 'df_tests_results.xlsx'
folder_tests_results   = "results/tests_results"
path_tests_results     = os.path.join(path_base, folder_tests_results, file_tests_results)

folder_tf_serving = "tf_serving"
tf_serving_path = os.path.join(path_base, folder_tf_serving)

results_path = Path('/Users/javi/Desktop/ML/ML_predictor/keras')

file_model_name = f'version01.keras'
path_keras = (results_path / file_model_name).as_posix()






