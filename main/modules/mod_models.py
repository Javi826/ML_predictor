#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 21:35:06 2024
@author: javi
"""
from functions.def_functions import class_weights
from keras.layers import Input, LSTM, concatenate, BatchNormalization, Dense
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from functions.def_functions import set_seeds
from paths.paths import results_path
import pandas as pd


def build_model(dropouts, n_neur1, n_neur2, n_neurd, le_rate, l2_regu, optimizers, lags, fets, n_features):
    
    
    array_dim = lags * fets
    print(n_features)
    
    #INPUT LAYERS
    #------------------------------------------------------------------------------
    input_lags   = Input(shape=(array_dim, n_features), name='input_Lags')
    input_months = Input(shape=(12,), name='input_Months')

    #LSTM LAYERS
    #------------------------------------------------------------------------------
    lstm_layer1 = LSTM(units=n_neur1, dropout=dropouts, name='LSTM1', return_sequences=True)(input_lags)
    lstm_layer2 = LSTM(units=n_neur2, dropout=dropouts, name='LSTM2')(lstm_layer1)
    #lstm_layer2 = LSTM(units=n_neur2, dropout=dropouts, name='LSTM2', return_sequences=True)(lstm_layer1)

    #CONCATENATE MODEL + BATCHNORMALIZATION
    #------------------------------------------------------------------------------
    merge_concatenat = concatenate([lstm_layer2, input_months])
    batch_normalized = BatchNormalization()(merge_concatenat)

    #DENSE LAYER
    #------------------------------------------------------------------------------
    dense_layer1 = Dense(n_neurd, activation='relu', kernel_regularizer=l2(l2_regu))(batch_normalized)
    #dense_layer2 = Dense(n_neurd, activation='relu', kernel_regularizer=l2(l2_regu))(dense_layer1)
    #dense_layer3 = Dense(n_neurd, activation='relu', kernel_regularizer=l2(l2_regu))(dense_layer2)
    output_layer = Dense(1,  activation='sigmoid', name='output')(dense_layer1)

    #MODEL DEFINITION + OPTIMIZER + COMPILE
    #------------------------------------------------------------------------------
    model       = Model(inputs=[input_lags, input_months], outputs=output_layer)
    optimizers  = Adam(learning_rate=le_rate)
    model.compile(optimizer=optimizers, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    #model.summary()
    
    
    return model

def train_model(model, X_train, y_train, X_valid, y_valid, dropout, batchsz, epochss, patient):

    file_model_name = f"dropout_{dropout}.keras"   
    path_keras      = (results_path / file_model_name).as_posix()
    
    set_seeds()
    
    class_weightss  = class_weights(y_train)
    
    check_pointers = ModelCheckpoint(filepath=path_keras, verbose=0, monitor='val_accuracy',mode='max',save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patient, verbose=0, restore_best_weights=True)

    history = model.fit(X_train, y_train, 
                        epochs=epochss, 
                        verbose=0,
                        batch_size=batchsz,
                        validation_data=(X_valid, y_valid),
                        #class_weight=class_weightss,
                        callbacks=[check_pointers, early_stopping])
    
    return history
