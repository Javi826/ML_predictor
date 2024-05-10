#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 21:35:06 2024
@author: javi
"""
from main.functions.def_functions import class_weights,set_seeds
from main.paths.paths import results_path
from keras.layers import Input, LSTM, concatenate, BatchNormalization, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout



def build_model(dropouts, n_neur1, n_neur2, n_neurd, le_rate, l2_regu, optimizers, lags,dim_arrays, n_features, d_layers):
    

    #INPUT LAYERS
    #------------------------------------------------------------------------------
    input_lags   = Input(shape=(dim_arrays, n_features), name='input_Lags')
    input_months = Input(shape=(12,), name='input_Months')
    
    #LSTM LAYERS
    #------------------------------------------------------------------------------
    lstm_layer1 = LSTM(units=n_neur1, dropout=dropouts, name='LSTM1', return_sequences=True)(input_lags)
    lstm_layer2 = LSTM(units=n_neur2, dropout=dropouts, name='LSTM2')(lstm_layer1)
    #lstm_layer2 = LSTM(units=n_neur2, name='LSTM2')(lstm_layer1)
    

    #CONCATENATE MODEL + BATCHNORMALIZATION
    #------------------------------------------------------------------------------
    merge_concatenat = concatenate([lstm_layer2, input_months])
    batch_normalized = BatchNormalization()(merge_concatenat)

    #DENSE LAYERS
    #-----------------------------------------------------------------------------
    dense_layer = batch_normalized
    
    for i in range(d_layers):
       dense_layer = Dense(n_neurd, activation='relu', kernel_regularizer=l2(l2_regu), name=f'Dense_{i+1}')(dense_layer)
       #dense_layer = Dropout(dropouts)(dense_layer)

    output_layer = Dense(1,  activation='sigmoid', name='output')(dense_layer)
    

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
    
    #class_weightss  = class_weights(y_train)
    
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
