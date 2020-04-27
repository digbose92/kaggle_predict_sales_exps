import os
import gc
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb 
import pickle

data = pd.read_pickle('../data/tot_data_new_v2.pkl')
test  = pd.read_csv('../data/test.csv').set_index('ID')
data.fillna(0,inplace=True)

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

xtrain=lgb.Dataset(X_train, label=Y_train)
xvalid=lgb.Dataset(X_valid, label=Y_valid)

params={'colsample_bytree': 0.75, 'learning_rate': 0.01, 'subsample':0.75, 'metric': 'rmse', 'min_data_in_leaf': 1024, 'bagging_seed': 128, 
               'num_leaves': 2048,'objective': 'regression', 'seed': 1204,'subsample': 0.75,'bagging_freq':1}

print('Predict using lightgbm')
lgb_model = lgb.train(params,xtrain,1000,xvalid,verbose_eval=True,early_stopping_rounds=100)

Y_valid_predict=lgb_model.predict(X_valid,lgb_model.best_iteration+1)
mean_sq_error=mean_squared_error(Y_valid,Y_valid_predict)
print('Validation RMSE:%f:'%(sqrt(mean_sq_error)))

Y_test_lgbm = lgb_model.predict(X_test,lgb_model.best_iteration+1).clip(0, 20)
predictions_lgbm={'Y_lgbm':Y_test_lgbm}

with open("Pred_check_lgbm_d_leaf_1024_n_leaves_2048.pkl","wb") as f:
    pickle.dump(predictions_lgbm,f)

#submission = pd.DataFrame({
#    "ID": test.index, 
#   "item_cnt_month": Y_test_lgbm
#})

#submission.to_csv('rf_submission_3.csv', index=False)

