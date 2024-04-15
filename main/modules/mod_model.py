#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 21:35:06 2024

@author: javi
"""

from keras.layers import Input, LSTM, Embedding, Reshape, concatenate, BatchNormalization, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from functions.def_functions import set_seeds


def build_model(dropout_ra, n_neur1_ra, n_neur2_ra, n_neur3_ra, le_rate_ra, l2_regu_ra, optimizers, lags, n_features):
    
    #INPUT LAYERS
    input_lags   = Input(shape=(lags, n_features), name='input_Lags')
    input_months = Input(shape=(12,), name='input_Months')
    input_days   = Input(shape=(1,), name='input_Days')

    #LSTM LAYERS
    lstm_layer1 = LSTM(units=n_neur1_ra, dropout=dropout_ra, name='LSTM1', return_sequences=True)(input_lags)
    lstm_layer2 = LSTM(units=n_neur2_ra, dropout=dropout_ra, name='LSTM2')(lstm_layer1)

    #EMBEDDINGS LAYER
    dweek_embedding = Embedding(input_dim=5, output_dim=5)(input_days)
    dweek_embedding = Reshape(target_shape=(5,))(dweek_embedding)

    #CONCATENATE MODEL + BATCHNORMALIZATION
    merge_concatenat = concatenate([lstm_layer2, input_months, dweek_embedding])
    batch_normalized = BatchNormalization()(merge_concatenat)

    #DENSE LAYER
    denses_layer = Dense(n_neur3_ra, activation='relu', kernel_regularizer=l2(l2_regu_ra))(batch_normalized)
    output_layer = Dense(1,  activation='sigmoid', name='output')(denses_layer)

    #MODEL DEFINITION + OPTIMIZER + COMPILE
    model      = Model(inputs=[input_lags, input_months, input_days], outputs=output_layer)
    optimizers  = Adam(learning_rate=le_rate_ra)
    model.compile(optimizer=optimizers, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    
    return model


def train_model(model, X_train, y_train, X_valid, y_valid, batchs_ra, epochs_ra, patien_ra, path_keras):
    
    set_seeds()
    
    check_pointers = ModelCheckpoint(filepath=path_keras, verbose=0, monitor='val_accuracy',mode='max',save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patien_ra, verbose=0, restore_best_weights=True)

    history = model.fit(X_train, y_train, 
                        epochs=epochs_ra, 
                        verbose=0,
                        batch_size=batchs_ra,
                        validation_data=(X_valid, y_valid),
                        callbacks=[check_pointers, early_stopping])
    
    return history