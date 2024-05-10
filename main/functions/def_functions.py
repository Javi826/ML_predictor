#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def day_week(df_build):
       
    df_build['date']     = pd.to_datetime(df_build['date'])
    df_build['day_week'] = df_build['date'].dt.dayofweek   
    
    return df_build

def date_anio(df_build):
    
    df_build['date']      = pd.to_datetime(df_build['date'])    
    df_build['date_anio'] = df_build['date'].dt.year.astype(str).str[:4]
    
    return df_build

def one_hot_months(df_build):
    
    df_build['date'] = pd.to_datetime(df_build['date'])  
    
    months_columns = []
    for month in range(1, 13):
        month_name = f'month_{month:02d}'
        df_build[month_name] = (df_build['date'].dt.month == month).astype(int)
        months_columns.append(month_name)
    
    return df_build

def add_index_column(df_build):
    
    df_build.insert(0, 'index_id', range(1, len(df_build) + 1))
    df_build['index_id'] = df_build['index_id'].apply(lambda x: f'{x:05d}')
       
    return df_build

def rounding_data(df_build):

    columns_to_round           = ['open', 'high', 'low', 'close', 'adj_close']
    df_build[columns_to_round] = df_build[columns_to_round].astype(float)
    df_build['day_week']       = df_build['day_week'].astype(int)
    
    for column in columns_to_round:
      if column in df_build.columns:
          df_build[column] = df_build[column].round(4)
            
    return df_build

def sort_columns(df_build):

    month_columns = [col for col in df_build.columns if col.startswith('month_')]    
    desired_column_order = ['index_id', 'date_anio', 'date', 'day_week', 'close', 'open', 'high', 'low', 'adj_close', 'volume'] + month_columns
    df_build = df_build[desired_column_order]
    
    return df_build

def diff_series(series, diff):
    
    diff = diff
    diff_series = series.diff(periods=diff)
    
    return diff_series

def filter_data_by_date_range(df, filter_start_date, filter_endin_date):
        
    return df[(df['date'] >= filter_start_date) & (df['date'] <= filter_endin_date)]

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def class_weights(y_train):
    
    c0, c1 = np.bincount(y_train)   
    w0 = (1/c0) * (len(y_train)) / 2
    w1 = (1/c1) * (len(y_train)) / 2
        
    return {0: w0, 1: w1}
      
def evaluate_history(lags, n_years_train, m_years_valid, start_train, start_valid, dropout, n_neur1, d_layers, n_neurd,batchsz, le_rate,l2_regu, optimizers, patient, history):

    ev_results = pd.DataFrame(history.history)
    ev_results.index += 1

    #BEST TRAIN Metrics
    best_train_loss       = ev_results['loss'].min()
    best_train_accu       = ev_results['accuracy'].max()
    best_train_AUCr       = ev_results['AUC'].max()
    best_train_epoch_loss = ev_results['loss'].idxmin()
    best_train_epoch_accu = ev_results['accuracy'].idxmax()
    best_train_epoch_AUCr = ev_results['AUC'].idxmax()
    
    #BEST VALID Metrics    
    best_valid_loss       = ev_results['val_loss'].min()
    best_valid_accu       = ev_results['val_accuracy'].max()
    best_valid_AUCr       = ev_results['val_AUC'].max()
    best_valid_epoch_loss = ev_results['val_loss'].idxmin()
    best_valid_epoch_accu = ev_results['val_accuracy'].idxmax()
    best_valid_epoch_AUCr = ev_results['val_AUC'].idxmax()

    #LAST Metrics
    last_train_loss = ev_results['loss'].iloc[-1]
    last_train_accu = ev_results['accuracy'].iloc[-1]
    last_train_AUCr = ev_results['AUC'].iloc[-1]
    last_valid_loss = ev_results['val_loss'].iloc[-1]
    last_valid_accu = ev_results['val_accuracy'].iloc[-1]
    last_valid_AUCr = ev_results['val_AUC'].iloc[-1]
    
    return {
        'Lags': lags,
        'n_years_train': n_years_train,
        'm_years_train': m_years_valid,
        'Start_train': start_train[0],
        'Start_valid': start_valid[0],
        'Dropout': dropout,
        'Neur_1': n_neur1,
        'Dense_layers':d_layers,
        'Neur_d': n_neurd,
        'Batch Size': batchsz,
        'Learning Rate': le_rate,
        'L2 Regurlar':l2_regu,
        'Optimizer': optimizers,
        'Patience': patient,
        'best_train_loss': best_train_loss,
        'best_train_accu': best_train_accu,
        'best_train_AUC': best_train_AUCr,
        'best_train_epoch_loss': best_train_epoch_loss,
        'best_train_epoch_accu': best_train_epoch_accu,
        'best_train_epoch_AUC': best_train_epoch_AUCr,
        'best_valid_loss': best_valid_loss,
        'best_valid_accu': best_valid_accu,
        'best_valid_AUC': best_valid_AUCr,
        'best_valid_epoch_loss': best_valid_epoch_loss,
        'best_valid_epoch_accu': best_valid_epoch_accu,
        'best_valid_epoch_AUC': best_valid_epoch_AUCr,
        'last_train_loss': last_train_loss,
        'last_train_accu': last_train_accu,
        'last_train_AUC': last_train_AUCr,
        'last_valid_loss': last_valid_loss,
        'last_valid_accu': last_valid_accu,
        'last_valid_AUC': last_valid_AUCr
    }

def print_results(ev_results):
    
   #print("best_valid_epoch_accu:", round(ev_results['best_valid_epoch_accu'], 2))
   #print("best_valid_epoch_AUC :", round(ev_results['best_valid_epoch_AUC'], 2))
   print("best_train_accu      :", round(ev_results['best_train_accu'], 2))
   print("best_valid_accu      :", round(ev_results['best_valid_accu'], 2))
   print("best_valid_AUC       :", round(ev_results['best_valid_AUC'], 2))
   print('\n')
    

def train_results(lags, n_years_train, m_years_valid, start_train, start_valid, dropout,n_neur1,d_layers, n_neurd, batchsz,le_rate,l2_regu, optimizers,patient,means_training_results):
    
    columns_mean = ['best_train_loss',       'best_train_accu',       'best_train_AUC',  'best_train_epoch_loss', 'best_train_epoch_accu',
                    'best_train_epoch_AUC',  'best_valid_loss',       'best_valid_accu', 'best_valid_AUC',        'best_valid_epoch_loss',
                    'best_valid_epoch_accu', 'best_valid_epoch_AUC', 'last_train_loss',  'last_train_accu',       'last_train_AUC',
                    'last_valid_loss',       'last_valid_accu',      'last_valid_AUC']
    
    df_means_training = pd.DataFrame(means_training_results)
    mean_values       = df_means_training[columns_mean].mean()  
    
    mean_best_train_loss       = mean_values['best_train_loss']
    mean_best_train_accu       = mean_values['best_train_accu']
    mean_best_train_AUC        = mean_values['best_train_AUC']
    mean_best_train_epoch_loss = mean_values['best_train_epoch_loss']
    mean_best_train_epoch_accu = mean_values['best_train_epoch_accu']
    mean_best_train_epoch_AUC  = mean_values['best_train_epoch_AUC']
    mean_best_valid_loss       = mean_values['best_valid_loss']
    mean_best_valid_accu       = mean_values['best_valid_accu']
    mean_best_valid_AUC        = mean_values['best_valid_AUC']
    mean_best_valid_epoch_loss = mean_values['best_valid_epoch_loss']
    mean_best_valid_epoch_accu = mean_values['best_valid_epoch_accu']
    mean_best_valid_epoch_AUC  = mean_values['best_valid_epoch_AUC']
    mean_last_train_loss       = mean_values['last_train_loss']
    mean_last_train_accu       = mean_values['last_train_accu']
    mean_last_train_AUC        = mean_values['last_train_AUC']
    mean_last_valid_loss       = mean_values['last_valid_loss']
    mean_last_valid_accu       = mean_values['last_valid_accu']
    mean_last_valid_AUC        = mean_values['last_valid_AUC']
    
    return {
        'Lags': lags,
        'n_years_train': n_years_train,
        'm_years_train': m_years_valid,
        'Start_train': start_train[0],
        'Start_valid': start_valid[0],
        'Dropout': dropout,
        'Neuro_1': n_neur1,
        'd_layers':d_layers,
        'Neuro_d': n_neurd,
        'Batch Size': batchsz,
        'Learning Rate': le_rate,
        'L2 Regurlar':l2_regu,
        'Optimizer': optimizers,
        'Patience': patient,
        'mean_best_train_loss': mean_best_train_loss,
        'mean_best_train_accu': mean_best_train_accu,
        'mean_best_train_AUC': mean_best_train_AUC,
        'mean_best_train_epoch_loss': mean_best_train_epoch_loss,
        'mean_best_train_epoch_accu': mean_best_train_epoch_accu,
        'mean_best_train_epoch_AUC': mean_best_train_epoch_AUC,
        'mean_best_valid_loss': mean_best_valid_loss,
        'mean_best_valid_accu': mean_best_valid_accu,
        'mean_best_valid_AUC': mean_best_valid_AUC,
        'mean_best_valid_epoch_loss': mean_best_valid_epoch_loss,
        'mean_best_valid_epoch_accu': mean_best_valid_epoch_accu,
        'mean_best_valid_epoch_AUC': mean_best_valid_epoch_AUC,
        'mean_last_train_loss': mean_last_train_loss,
        'mean_last_train_accu': mean_last_train_accu,
        'mean_last_train_AUC': mean_last_train_AUC,
        'mean_last_valid_loss': mean_last_valid_loss,
        'mean_last_valid_accu': mean_last_valid_accu,
        'mean_last_valid_AUC': mean_last_valid_AUC
    }

def tests_results(rets,lags, n_years_train, m_years_valid, start_tests, endin_tests, dropout,n_neur1,d_layers, n_neurd, batchsz,le_rate,l2_regu, optimizers,patient,tests_accuracy):
    

    return {
        'Rets': rets,
        'Lags': lags,
        'n_years_train': n_years_train,
        'm_years_train': m_years_valid,
        'Start_tests': start_tests[0],
        'endin_tests': endin_tests[0],
        'Dropout': dropout,
        'Neuro_1': n_neur1,
        'd_layers':d_layers,
        'Neuro_d': n_neurd,
        'Batch Size': batchsz,
        'Learning Rate': le_rate,
        'L2 Regurlar':l2_regu,
        'Optimizer': optimizers,
        'Patience': patient,
        'tests_accuracy': tests_accuracy,
 
    }


def plots_loss(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    epochs = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

def plots_accu(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.plot(epochs, (history.history['accuracy']), label='Training Accuracy')
    plt.plot(epochs, (history.history['val_accuracy']), label='Validation Accuracy ')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
def plots_aucr(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    epochs = range(1, len(history.history['AUC']) + 1)
    plt.plot(epochs, history.history['AUC'], label='Training AUC')
    plt.plot(epochs, history.history['val_AUC'], label='Validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()

def df_plots(x, y, x_label, y_label,plot_style):
    
    plt.figure(figsize=(10, 6))
    
    if plot_style == "lines":
        plt.plot(x, y, label=f'{x_label} vs {y_label}')  # Use a line plot
    elif plot_style == "points":
        plt.scatter(x, y, label=f'{x_label} vs {y_label}', marker='o')  # Use a scatter plot with markers
    else:
        raise ValueError("Invalid plot_style. Use 'lines' or 'points'.")

    plt.title(f'{x_label} vs {y_label} Plot')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()
            
def plots_histograms(dataframe, columns_of_interest):
    bins = 30
    figsize = (15, 5)
    
    # plot
    fig, axes = plt.subplots(nrows=1, ncols=len(columns_of_interest), figsize=figsize)
    
    # Columns
    for i, column in enumerate(columns_of_interest):
        axes[i].hist(dataframe[column], bins=bins, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Histogram de {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('frequency')
    
    # design
    plt.tight_layout()
    plt.show()
    
    
def time_intervals(df_preprocess, n_years_train, m_years_valid, endin_train):
    
    endin_train = pd.to_datetime(endin_train[0]) 

    start_date = df_preprocess['date'].min()
    endin_date = endin_train

    train_intervals = []
    while start_date < endin_date:
        
        endin_train = start_date.replace(year=start_date.year + n_years_train)
        start_valid = endin_train
        endin_valid = start_valid.replace(year=start_valid.year + m_years_valid)
        
        if endin_valid > endin_date: endin_valid = endin_date

        start_date_str  = start_date.strftime('%Y-%m-%d')
        endin_train_str = endin_train.strftime('%Y-%m-%d')
        start_valid_str = start_valid.strftime('%Y-%m-%d')
        endin_valid_str = endin_valid.strftime('%Y-%m-%d')

        train_intervals.append((start_date_str, endin_train_str, start_valid_str, endin_valid_str))

        start_date = endin_valid

    return train_intervals