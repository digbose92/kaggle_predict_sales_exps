import pandas as pd
import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb

seed=1204
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

def score(params):
    print("Training with params:")
    print(params)
    lgb_model = lgb.train(params,xtrain,1000,xvalid,verbose_eval=True,early_stopping_rounds=100)
    predictions = lgb_model.predict(X_valid,ntree_limit=lgb_model.best_iteration + 1)
    rmse_loss=mean_squared_error(predictions,Y_valid)
    rmse_loss=sqrt(rmse_loss)
    return {'loss': rmse_loss, 'status': STATUS_OK}

def optimize(space,seed=seed,max_evals=5):
    
    best = fmin(score, space, algo=tpe.suggest, 
        # trials=trials, 
        max_evals=max_evals)
    return best
    
space={'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'min_data_in_leaf': hp.choice('min_data_in_leaf',[1024,1152,1280,1408,2048]),
    'num_leaves' : hp.choice('num_leaves',[1024,1152,1280,1408,2048]),
    'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
    'seed':seed,
    'objective': 'regression',
    'silent': 1,
    'metric':'rmse'}

best_hyperparams = optimize(space,max_evals=250)
print("The best hyperparameters are: ")
print(best_hyperparams)

