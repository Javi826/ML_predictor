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
    df_recove_data['returns']    = tests_data['returns']
    df_recove_data['market_ret'] = (tests_data['close'].pct_change())
    df_recove_data['returns']    = tests_data['returns']
   
    df_predictions = pd.DataFrame({'y_tests': y_tests,'y_pred_bin': y_pred_bin})    
    df_recove_data.reset_index(drop=True, inplace=True)
    df_predictions.reset_index(drop=True, inplace=True)
    
    df_signals                = pd.concat([df_recove_data, df_predictions], axis=1)
    #df_backtesting ['actual_ret'] = df_backtesting['market_ret'] * df_backtesting['y_tests'].shift(1)
    #df_backtesting ['strate_ret'] = df_backtesting['market_ret'] * df_backtesting['y_pred_bin'].shift(1)
    
    #SIGNAL
    #---------------------------------------------------------------------------------------------------   
    column_choose = 'y_pred_bin'
    
    total_sum_next_five = df_signals[column_choose].iloc[1:6].sum()
    print(total_sum_next_five)
    
    if total_sum_next_five > 2:
       df_signals.loc[0, column_choose] = 0
    
    
    df_signals['signal'] = np.select(
        [(df_signals[column_choose] == 1) & (df_signals[column_choose].shift(-1) == 0),
         (df_signals[column_choose] == 1) & (df_signals[column_choose].shift(-1) == 1),
         (df_signals[column_choose] == 0) & (df_signals[column_choose].shift(-1) == 0), 
         (df_signals[column_choose] == 0) & (df_signals[column_choose].shift(-1) == 1)], 
        [-1, 0, 0, 1], default=np.nan)

      
    # SAVE FILE
    #---------------------------------------------------------------------------------------------------
    excel_signals_path = os.path.join(path_base, folder_tests_results, f"df_market_signals_{start_tests_i}.xlsx")
    df_signals.to_excel(excel_signals_path, index=False) 
    
    #BACKTESTING
    #---------------------------------------------------------------------------------------------------
    
    df_backtesting = df_signals[df_signals['signal'] != 0].copy()
    
    df_backtesting.loc[df_backtesting['signal'] ==  1, 'oper_rent'] = 0
    df_backtesting.loc[df_backtesting['signal'] == -1, 'oper_rent'] = df_backtesting['close'].pct_change()
    
    comision = 0.1
    
    df_backtesting.loc[df_backtesting['oper_rent'] != 0, 'oper_rent_gross'] = df_backtesting['oper_rent']
    df_backtesting.loc[df_backtesting['oper_rent'] != 0, 'oper_rent_nets']  = df_backtesting['oper_rent'] - (comision/100)
    
    initial_capital = 10000
    df_backtesting['gross_capital'] = initial_capital
    df_backtesting['nets_capital']  = initial_capital
    
    
    df_backtesting['gross_capital'] = (df_backtesting['oper_rent_gross'] + 1).cumprod() * initial_capital
    df_backtesting['nets_capital']  = (df_backtesting['oper_rent_nets']  + 1).cumprod() * initial_capital
    
    last_capital = df_backtesting['nets_capital'].dropna().iloc[-1]
    
    percentage_change = ((last_capital - initial_capital) / initial_capital) * 100
    
    price_first           = tests_data['close'].iloc[0]
    price_lasts           = tests_data['close'].iloc[-1]
    total_return_buy_hold = (price_lasts - price_first) / price_first * 100
    n_operations = df_backtesting.shape[0]
    
    # Imprimir los valores con dos decimales
    print("Initl capital value  :", "{:.2f}".format(initial_capital))
    print("Final capital value  :", "{:.2f}".format(last_capital))
    print("Rentability buy&hold :", "{:.2f}".format(total_return_buy_hold), "%")
    print("Rentability strategy :", "{:.2f}".format(percentage_change), "%")
    print("Number of operations :", n_operations)


    
    excel_back_path = os.path.join(path_base, folder_tests_results, f"df_backtesting_{start_tests_i}.xlsx")
    df_backtesting.to_excel(excel_back_path, index=False)

        
    return

