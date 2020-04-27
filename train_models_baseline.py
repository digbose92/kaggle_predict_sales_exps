import os
import gc
import pickle
import time
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_pickle('../data/tot_data_updated.pkl')
print(data.columns)
test  = pd.read_csv('../data/test.csv').set_index('ID')
#data['item_price_sum']=data['item_price_sum'].astype(np.float16)
#data['item_mean_sum']=data['item_mean_sum'].astype(np.float16)
data.fillna(0,inplace=True)
print(len(data.columns))
print(data.columns)

data = data[[
   'date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'shop_city',
       'shop_category', 'item_category_id','item_cnt_month_lag_1','item_cnt_month_lag_2',
       'item_cnt_month_lag_3', 'date_avg_item_cnt_lag_1',
       'date_item_avg_item_cnt_lag_1', 'date_item_avg_item_cnt_lag_2',
       'date_item_avg_item_cnt_lag_3', 'date_shop_avg_item_cnt_lag_1',
       'date_shop_avg_item_cnt_lag_2', 'date_shop_avg_item_cnt_lag_3',
       'date_cat_avg_item_cnt_lag_1', 'date_shop_cat_avg_item_cnt_lag_1','date_city_avg_item_cnt_lag_1',
       'date_item_city_avg_item_cnt_lag_1','item_avg_item_price',
       'date_item_avg_item_price_lag_1',
       'date_item_avg_item_price_lag_2', 'date_item_avg_item_price_lag_3',
#     'date_type_avg_item_cnt_lag_1',
#        'date_subtype_avg_item_cnt_lag_1', 
    'date_shop_item_avg_item_cnt_lag_1',
    'delta_revenue_lag_1', 'month', 'days', 'item_shop_first_sale', 'item_first_sale']]


"""data = data[[
   'date_block_num', 'shop_id', 'item_id', 'item_cnt_month','item_category_id','item_cnt_month_lag_1','item_cnt_month_lag_2',
       'item_cnt_month_lag_3','date_avg_item_cnt_lag_1',
       'date_item_avg_item_cnt_lag_1','date_item_avg_item_cnt_lag_2',
       'date_item_avg_item_cnt_lag_3','date_shop_avg_item_cnt_lag_1',
       'date_shop_avg_item_cnt_lag_2','date_shop_avg_item_cnt_lag_3',
       'date_cat_avg_item_cnt_lag_1','date_shop_cat_avg_item_cnt_lag_1',
       'date_shop_subtype_avg_item_cnt_lag_1','date_city_avg_item_cnt_lag_1',
       'date_item_city_avg_item_cnt_lag_1', 
#     'date_type_avg_item_cnt_lag_1',
#        'date_subtype_avg_item_cnt_lag_1', 
    'date_shop_item_avg_item_cnt_lag_1',
    'delta_price_lag','delta_revenue_lag_1', 'month', 'days', 'item_shop_first_sale', 'item_first_sale']]"""


#data=data[['date_block_num','shop_id',]]


X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
#data.fi
#del data 

print(X_train.shape)
print(Y_train.shape)
print(X_valid.shape)


model = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1,
    tree_method='gpu_hist',
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 20)

Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('xgb_submission_baseline_updated_feats_price.csv', index=False)

# save predictions for an ensemble
#pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))
#pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))"""

"""rf_model = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=0, n_jobs=-1, verbose=1)
rf_model.fit(X_train, Y_train)
rf_val_pred = rf_model.predict(X_valid)
valid_rmse=sqrt(mean_squared_error(Y_valid,rf_val_pred))
print('Validation RMSE:%f' %(valid_rmse))"""

