#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:42:47 2024
@author: javi
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main.paths.paths import path_base,folder_tests_results

def mod_backtesting(tests_data,y_tests,y_pred_bin,start_tests_i, endin_tests_i):

    df_recove_data = pd.DataFrame()
    df_recove_data['date']       = tests_data['date']
    df_recove_data['close']      = tests_data['close']
    df_recove_data['open']       = tests_data['open']
    df_recove_data['market_ret'] = np.log(tests_data['close'].pct_change())
   
    df_predictions = pd.DataFrame({'y_tests': y_tests,'y_pred_bin': y_pred_bin})    
    df_recove_data.reset_index(drop=True, inplace=True)
    df_predictions.reset_index(drop=True, inplace=True)
    
    df_backtesting                = pd.concat([df_recove_data, df_predictions], axis=1)
    df_backtesting ['actual_ret'] = df_backtesting['market_ret'] * df_backtesting['y_tests'].shift(1)
    df_backtesting ['strate_ret'] = df_backtesting['market_ret'] * df_backtesting['y_pred_bin'].shift(1) 
    
    # SAVE FILE
    excel_back_path = os.path.join(path_base, folder_tests_results, f"y_backtesting_{start_tests_i}.xlsx")
    df_backtesting.to_excel(excel_back_path, index=False)    

    return

############################################################################################################
    # Plot del histograma de la suma acumulativa
    cumulative_returns = df_backtesting[['strate_ret', 'actual_ret']].cumsum()
    cumulative_returns.hist(bins=50, figsize=(10, 6))
    plt.title('Histogram of Cumulative Returns')
    plt.xlabel('Cumulative Returns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    # Plot the cumulative sum over time
    cumulative_returns.plot(figsize=(10, 6))
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Time Index')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.show()
    
    # Calculate total accumulated return
    total_return_strategy = df_backtesting['strate_ret'].sum()
    total_return_actual   = df_backtesting['actual_ret'].sum()
    
    # Calculate mean return per time period
    mean_return_strategy = df_backtesting['strate_ret'].mean()
    mean_return_actual   = df_backtesting['actual_ret'].mean()
    
    # Calculate volatility
    volatility_strategy = df_backtesting['strate_ret'].std()
    volatility_actual   = df_backtesting['actual_ret'].std()
    
    # Assuming risk-free rate is 0 for simplicity
    risk_free_rate = 0.0
    
    # Calculate Sharpe ratio
    sharpe_ratio_strategy = (mean_return_strategy - risk_free_rate) / volatility_strategy
    sharpe_ratio_actual   = (mean_return_actual - risk_free_rate) / volatility_actual
    
    # Calculate max drawdown
    max_drawdown_strategy = (df_backtesting['strate_ret'].cumsum() - df_backtesting['strate_ret'].cumsum().cummax()).min()
    max_drawdown_actual   = (df_backtesting['actual_ret'].cumsum() - df_backtesting['actual_ret'].cumsum().cummax()).min()
    
    # Calculate Calmar ratio
    calmar_ratio_strategy = total_return_strategy / abs(max_drawdown_strategy)
    calmar_ratio_actual   = total_return_actual / abs(max_drawdown_actual)
    
    # Print results
     
    print('\n')
    print("Total accumulated return for strategy   :", round(total_return_strategy * 100, 2))
    print("Total accumulated return for actual     :", round(total_return_actual * 100, 2))
    print("Mean return per time period for strategy:", round(mean_return_strategy * 100, 2))
    print("Mean return per time period for actual  :", round(mean_return_actual * 100, 2))
    print("Volatility for strategy                 :", round(volatility_strategy, 2))
    print("Volatility for actual                   :", round(volatility_actual, 2))
    print("Sharpe ratio for strategy               :", round(sharpe_ratio_strategy * 100, 2))
    print("Sharpe ratio for actual                 :", round(sharpe_ratio_actual * 100, 2))
    print("Max drawdown for strategy               :", round(max_drawdown_strategy, 2))
    print("Max drawdown for actual                 :", round(max_drawdown_actual, 2))
    print("Calmar ratio for strategy               :", round(calmar_ratio_strategy, 2))
    print("Calmar ratio for actual                 :", round(calmar_ratio_actual, 2))
    
    price_first = tests_data['close'].iloc[0]
    price_lasts = tests_data['close'].iloc[-1]
    
    print(price_first)
    print(price_lasts)
    
    # Calcular el retorno acumulado para la estrategia de "comprar y mantener"
    total_return_buy_and_hold = (price_lasts - price_first) / price_first * 100
    
    # Mostrar el retorno acumulado
    print("Total accumulated return for Buy and Hold strategy:", total_return_buy_and_hold)