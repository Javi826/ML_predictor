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
    
def class_weight(df_preprocessing):
    c0, c1 = np.bincount(df_preprocessing['direction'])
    print(f"Clase 0: {c0} muestras, Clase 1: {c1} muestras")
    
    w0 = (1/c0) * (len(df_preprocessing)) / 2
    w1 = (1/c1) * (len(df_preprocessing)) / 2
    
    print(f"Peso de clase 0: {w0}, Peso de clase 1: {w1}")
    
    return {0: w0, 1: w1}


#def class_weight(df_preprocessing):
 #   
  #  c0, c1 = np.bincount(df_preprocessing['direction'])
#    w0 = (1/c0) * (len(df_preprocessing)) / 2
 #   w1 = (1/c1) * (len(df_preprocessing)) / 2
  #  return {0: w0, 1:w1}
  
    
def evaluate_history(lags, n_years_train, m_years_valid, start_train, start_valid, dropout, n_neur1_ra, batchs_ra, le_rate_ra, optimizers, patien_ra, history):

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
        'Neurons': n_neur1_ra,
        'Batch Size': batchs_ra,
        'Learning Rate': le_rate_ra,
        'Optimizer': optimizers,
        'Patience': patien_ra,
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
    
   print("best_valid_epoch_accu:", round(ev_results['best_valid_epoch_accu'], 2))
   print("best_valid_epoch_AUC :", round(ev_results['best_valid_epoch_AUC'], 2))
   print("best_valid_accu      :", round(ev_results['best_valid_accu'], 2))
   print("best_valid_AUC       :", round(ev_results['best_valid_AUC'], 2))

    
def create_results_df(lags, n_years_train, m_years_valid, start_train, start_valid, dropout, n_neurons_1, batch_s, le_rate, optimizers, patiences, ev_results):
    df_tra_val_results = pd.DataFrame({
        'Lags': [lags],
        'n_years_train': [n_years_train],
        'm_years_train': [m_years_valid],
        'Start_train': [start_train],
        'Start_valid': [start_valid],
        'Dropout': [dropout],
        'Neurons': [n_neurons_1],
        'Batch Size': [batch_s],
        'Learning Rate': [le_rate],
        'Optimizer': [optimizers],
        'Patience': [patiences],
        'Last train_Loss': [ev_results['last_train_loss']],
        'Last valid_Loss': [ev_results['last_valid_loss']],
        'Last train_accuracy': [ev_results['last_train_accu']],
        'Last valid_accuracy': [ev_results['last_valid_accu']],
        'Best train_accuracy': [ev_results['best_train_accu']],
        'Best valid_accuracy': [ev_results['best_valid_accu']],
        'Best train_epoch': [ev_results['best_train_epoch_accu']],
        'Best valid_epoch': [ev_results['best_valid_epoch_accu']],
        'Best train_AUC': [ev_results['best_train_AUC']],
        'Best valid_AUC': [ev_results['best_valid_AUC']]
    })
    return df_tra_val_results

def cross_training(cross_training_results):

    df_cross_training = pd.DataFrame(cross_training_results)
    
    columns_mean = [
        'best_train_loss', 'best_train_accu', 'best_train_AUC',
        'best_train_epoch_loss', 'best_train_epoch_accu', 'best_train_epoch_AUC',
        'best_valid_loss', 'best_valid_accu', 'best_valid_AUC',
        'best_valid_epoch_loss', 'best_valid_epoch_accu', 'best_valid_epoch_AUC',
        'last_train_loss', 'last_train_accu', 'last_train_AUC',
        'last_valid_loss', 'last_valid_accu', 'last_valid_AUC'
    ]

    mean_values = df_cross_training[columns_mean].mean()

    df_cross_training = pd.concat([df_cross_training, mean_values.to_frame().T], ignore_index=True)
    print("Mean results:")
    print(mean_values)

    return df_cross_training

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
    
    
def time_intervals(df_preprocess, n_years_train, m_years_valid):

    start_date = df_preprocess['date'].min()
    endin_date = df_preprocess['date'].max()

    time_intervals = []
    while start_date < endin_date:
        
        endin_train = start_date.replace(year=start_date.year + n_years_train)
        start_valid = endin_train
        endin_valid = start_valid.replace(year=start_valid.year + m_years_valid)
        
        if endin_valid > endin_date: endin_valid = endin_date

        start_date_str  = start_date.strftime('%Y-%m-%d')
        endin_train_str = endin_train.strftime('%Y-%m-%d')
        start_valid_str = start_valid.strftime('%Y-%m-%d')
        endin_valid_str = endin_valid.strftime('%Y-%m-%d')

        time_intervals.append((start_date_str, endin_train_str, start_valid_str, endin_valid_str))

        start_date = endin_valid

    return time_intervals