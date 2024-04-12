# DATASET CLEANING
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""
import os
from functions.def_functions import add_index_column,date_anio,day_week,sort_columns,rounding_data
from paths.paths import path_base,folder_df_build

def mod_data_build(df_data,start_date,endin_date):
    print('\n')
    print(f'START MODUL mod_data_build')
    
    #Restart dataframe jic
    restart_dataframes = True  
    if 'df_build' in locals() and restart_dataframes:del df_build  # delete dataframe if exits 
            
    df_build   = df_data.copy()
    df_build   = add_index_column(df_build)
    df_build   = day_week(df_build)
    df_build   = date_anio(df_build)
    df_build   = sort_columns(df_build)
    df_build   = rounding_data(df_build)
        
    # SAVE FILE with start_date and endin_date suffixes

    file_df_build = f"df_build_{start_date}_{endin_date}.csv"
    excel_file_path = os.path.join(path_base, folder_df_build, file_df_build)
    df_build.to_csv(excel_file_path, index=False)
    
    print(f'ENDIN MODUL mod_data_build\n')
    return df_build

