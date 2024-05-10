#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""
import os
from main.functions.def_functions import add_index_column,date_anio,day_week,sort_columns,rounding_data,one_hot_months
from main.paths.paths import path_base,folder_df_build

def mod_data_build(df_data,start_date,endin_date):
    
    #Restart dataframe jic
    restart_dataframes = True  
    if 'df_build' in locals() and restart_dataframes:del df_build  # delete dataframe if exits 
            
    df_build = df_data.copy()
    df_build = day_week(df_build)
    df_build = date_anio(df_build)
    df_build = one_hot_months(df_build)
    df_build = add_index_column(df_build)
    df_build = rounding_data(df_build)
    df_build = sort_columns(df_build)

    file_df_build = f"df_build_{start_date}_{endin_date}.csv"
    excel_file_path = os.path.join(path_base, folder_df_build, file_df_build)
    df_build.to_csv(excel_file_path, index=False)
    
    return df_build