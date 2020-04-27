import os
import gc
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.svm import SVR

data = pd.read_pickle('../data/tot_data_new_v2.pkl')
test  = pd.read_csv('../data/test.csv').set_index('ID')
data.fillna(0,inplace=True)
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

print('Data loading started')
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
print('Training SVR')

clf=SVR(kernel='rbf',verbose=True)
clf.fit(X_train,Y_train)
valid_predict=clf.predict(X_valid).clip(0, 20)


mse_3=mean_squared_error(valid_predict,Y_valid)
print('Validation RMSE:%f:'%(sqrt(mse_3)))
