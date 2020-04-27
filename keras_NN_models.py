import numpy as np
import pandas as pd
import _pickle as cPickle
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Activation
#from keras.layers.core import Dense, Dropout,Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import time
from keras.optimizers import Adam,SGD

def _build_keras_model(n_feats):
    
    print ('Creating simple nn model')
    model = Sequential()
    model.add(Dense(n_feats, input_dim=n_feats, kernel_initializer='normal', activation='relu'))
    #model.add(Dropout(0.02))
    model.add(Dense(512, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    model.add(Dense(1024, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    #model.add(Dense(4096, kernel_initializer='normal'))
    #model.add(BatchNormalization())
    #model.add(Activation("relu"))
    model.add(Dense(1024,kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    model.add(Dense(512, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    model.add(Dense(1, kernel_initializer='normal'))
    return(model)

def preprocess_nn_features_for_simple(data):
    #data here just train & validation 
    #scaler=StandardScaler(copy=True,with_mean=True,with_std=True) #0 mean and standard deviation-1
    scaler = MinMaxScaler()
    float_cols=['item_cnt_month_lag_1', 'item_cnt_month_lag_2',
        'item_cnt_month_lag_3', 'date_avg_item_cnt_lag_1',
        'date_item_avg_item_cnt_lag_1', 'date_item_avg_item_cnt_lag_2',
        'date_item_avg_item_cnt_lag_3', 'date_shop_avg_item_cnt_lag_1',
        'date_shop_avg_item_cnt_lag_2', 'date_shop_avg_item_cnt_lag_3',
        'date_shop_item_avg_item_cnt_lag_1',
        'date_shop_item_avg_item_cnt_lag_2',
        'date_shop_item_avg_item_cnt_lag_3',
        'date_shop_subtype_avg_item_cnt_lag_1', 'date_city_avg_item_cnt_lag_1',
        'date_item_city_avg_item_cnt_lag_1', 'delta_price_lag']
    data[float_cols]=scaler.fit_transform(data[float_cols])
    int_8_cols=['date_block_num', 'shop_id', 'shop_category', 'shop_city',
        'item_category_id', 'name2', 'subtype_code', 'type_code', 'month',
        'days', 'item_shop_first_sale', 'item_first_sale']
    int_16_cols=['item_id','name3']
    #add one hot encoding on the integer columns later
    return(data)

def run(X_train,Y_train,X_valid,Y_valid,n_feats=32):
    model=_build_keras_model(n_feats=n_feats)
    #Y_train_scaler=MinMaxScaler()
    #Y_valid_scaler=MinMaxScaler()
    #Y_train=Y_train_scaler.fit_transform(Y_train.reshape(-1,1))
    #Y_valid=Y_valid_scaler.fit_transform(Y_valid.reshape(-1,1))
    # Compile model
    adam=Adam(learning_rate=0.001)
    sgd=SGD(lr=0.1)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    print(model.summary())
    
    early_stopping=EarlyStopping(monitor='val_loss', patience=10)
    check_point = ModelCheckpoint('simple_weights.best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    fit_params={
        'epochs':100,
        'batch_size':64,
        'validation_data':(X_valid,Y_valid),
        'callbacks':[early_stopping, check_point],
        'shuffle':True
    }
    with tf.device('/gpu:0'):
        model.fit(X_train,Y_train,**fit_params)

if __name__ == "__main__":
    data=pd.read_pickle('../data/tot_data_new_v2.pkl')
    data_tot = data[data.date_block_num <= 33].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month'].values
    Y_valid = data[data.date_block_num == 33]['item_cnt_month'].values

    X_test=data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    """cols_select=['item_cnt_month_lag_1', 'item_cnt_month_lag_2',
        'item_cnt_month_lag_3', 'date_avg_item_cnt_lag_1',
        'date_item_avg_item_cnt_lag_1', 'date_item_avg_item_cnt_lag_2',
        'date_item_avg_item_cnt_lag_3', 'date_shop_avg_item_cnt_lag_1',
        'date_shop_avg_item_cnt_lag_2', 'date_shop_avg_item_cnt_lag_3',
        'date_shop_item_avg_item_cnt_lag_1',
        'date_shop_item_avg_item_cnt_lag_2',
        'date_shop_item_avg_item_cnt_lag_3',
        'date_shop_subtype_avg_item_cnt_lag_1', 'date_city_avg_item_cnt_lag_1',
        'date_item_city_avg_item_cnt_lag_1', 'delta_price_lag','date_block_num','item_shop_first_sale', 'item_first_sale']"""
    data_tot=preprocess_nn_features_for_simple(data_tot)
    #data_tot=data_tot[cols_select]

    #X_test=X_test[cols_select] 
    X_train = data_tot[data_tot.date_block_num < 33]
    X_valid = data_tot[data_tot.date_block_num == 33]
    
    run(X_train,Y_train,X_valid,Y_valid)






