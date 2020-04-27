import os
import gc
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
import xgboost as xgb


data = pd.read_pickle('../data/tot_data_new_v2.pkl')
test  = pd.read_csv('../data/test.csv').set_index('ID')
data.fillna(0,inplace=True)

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

#{'booster': 'gbtree', 'colsample_bytree': 0.55, 'eta': 0.1, 'eval_metric': 'rmse', 'gamma': 0.8, 'max_depth': 11, 'min_child_weight': 6.0, 'n_estimators': 914.0, 'nthread': 45, 'objective': 'reg:squarederror', 'seed': 314159265, 'silent': 1, 'subsample': 0.65, 'tree_method': 'exact'}

#train xgboost model here
print('Training xgboost model here')
"""model = XGBRegressor(
    gamma=0.8,
    max_depth=11,
    n_estimators=914,
    min_child_weight=6.0, 
    colsample_bytree=0.55, 
    subsample=0.65,
    tree_method='exact',
    objective='reg:squarederror',
    eta=0.1,
    seed=31415926
    )

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train),(X_valid, Y_valid)], 
    verbose=True,
    early_stopping_rounds = 20)"""
params={'booster': 'gbtree', 'colsample_bytree': 0.55, 'eta': 0.1, 'eval_metric': 'rmse', 'gamma': 0.8, 'max_depth': 11, 'min_child_weight': 6.0, 'nthread': 45, 'objective': 'reg:squarederror', 'seed': 314159265, 'silent': 1, 'subsample': 0.65, 'tree_method': 'exact'}
dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_valid, label=Y_valid)
watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
num_round=914
model = xgb.train(params, dtrain, num_round,
                          evals=watchlist,
                          verbose_eval=True,
                          early_stopping_rounds = 20)
X_test_gb=xgb.DMatrix(X_test)
Y_test_xgb = model.predict(X_test_gb,ntree_limit=model.best_iteration + 1).clip(0, 20)

#train random forest model here
print('Training random forest model here')
regr_3 = RandomForestRegressor(n_estimators=500,max_depth=30,min_samples_split=5,min_samples_leaf=4,bootstrap=True,max_features="sqrt",verbose=1,n_jobs=45)
regr_3.fit(X_train,Y_train)
valid_predict_3=regr_3.predict(X_valid).clip(0, 20)

mse_3=mean_squared_error(valid_predict_3,Y_valid)
print('Validation RMSE2:%f:'%(sqrt(mse_3)))

print('Test prediction using random forest')
#predict using regr_2 on X_test
Y_test_rf = regr_3.predict(X_test).clip(0, 20)



print('Ensembling by simple averaging')
Y_test_ensemble_v1=(0.7*(Y_test_rf)+0.3*(Y_test_xgb))

predictions={'Y_rf':Y_test_rf,'Y_xgb':Y_test_xgb}

with open("Pred_check.pkl","wb") as f:
    pickle.dump(predictions,f)


submission = pd.DataFrame({
    "ID": test.index, 
   "item_cnt_month": Y_test_ensemble_v1
})
submission.to_csv('ensemble_submission_1_new_weighted.csv', index=False)


