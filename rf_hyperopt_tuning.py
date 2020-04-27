import pandas as pd
import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor

seed=1204
data = pd.read_pickle('../data/tot_data_new_v2.pkl')
test  = pd.read_csv('../data/test.csv').set_index('ID')
data.fillna(0,inplace=True)
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

def score(params):
    print("Training with params:")
    print(params)
    regr_rf = RandomForestRegressor(**params,)
    regr_rf.fit(X_train,Y_train)
    valid_predict=regr_rf.predict(X_valid).clip(0, 20)
    mse_loss=mean_squared_error(valid_predict,Y_valid)
    rmse_loss=sqrt(mse_loss)
    return {'loss': rmse_loss, 'status': STATUS_OK}

def optimize(space,seed=seed,max_evals=5):
    
    best = fmin(score, space, algo=tpe.suggest, 
        # trials=trials, 
        max_evals=max_evals)
    return best

space={'max_depth': hp.choice('max_depth', np.arange(30,55,5)),
    'min_samples_split': hp.choice('min_samples_split', np.arange(5,11,1)),
    'min_samples_leaf': hp.choice('min_data_in_leaf',np.arange(4,10,1)),
    'bootstrap':True,
    'max_features':"sqrt",
    'verbose':1,
    'n_jobs':30,
    'n_estimators':500}

best_hyperparams = optimize(space,max_evals=250)
print("The best hyperparameters for random forest are: ")
print(best_hyperparams)
