import os
import gc
import pickle
import time
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle

data = pd.read_pickle('../data/tot_data_new_v2.pkl')
test  = pd.read_csv('../data/test.csv').set_index('ID')
data.fillna(0,inplace=True)


eta_range=[0.02,0.03,0.04,0.05]
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid= data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid= data[data.date_block_num == 33]['item_cnt_month']
dtrain=xgb.DMatrix(X_train,label=Y_train)
dvalid=xgb.DMatrix(X_valid,label=Y_valid)
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)


num_boost_round=1000
min_rmse = float("Inf")
min_child_weight_range=[0.5,1.0,1.5]
max_depth_range=[6,8,10,12,14]
dict_results=dict()
eval_dict_num=0
for m_c_range in min_child_weight_range:
    for m_d_range in max_depth_range:
        print("Weight:%f,Depth:%f"%(m_c_range,m_d_range))
        dict_temp=dict()
        model = XGBRegressor(max_depth=m_d_range,
        n_estimators=num_boost_round,min_child_weight=m_c_range,colsample_bytree=0.5,subsample=0.5,tree_method='exact',objective='reg:squarederror',eta=0.01)
        #print("Validation with ={}".format(eta))
    #params['eta']=eta
        model.fit(
            X_train, 
            Y_train, 
            eval_metric="rmse", 
            eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
            verbose=True,
            early_stopping_rounds = 10)

        best_iter=model.get_booster().best_iteration
        Y_pred = model.predict(X_valid,ntree_limit=best_iter).clip(0, 20)
        mse=mean_squared_error(Y_valid,Y_pred)
        rmse=sqrt(mse)
        dict_temp['Depth']=m_d_range
        dict_temp['Weight']=m_c_range
        dict_temp['rmse']=rmse
        dict_results[str(eval_dict_num)]=dict_temp
        eval_dict_num=eval_dict_num+1
        print("Weight:%f,Depth:%f,RMSE:%f"%(m_c_range,m_d_range,rmse))
        if(rmse < min_rmse):
            min_rmse=rmse
            best_weight=m_c_range
            best_depth=m_d_range
            print('Best parameters(Weight,Depth) till now:%f,%f'%(best_weight,best_depth))

#print(dict_temp)  
with open("Best_param.pkl","wb") as f:
    pickle.dump(dict_results,f)

    #print(dict_temp)
    #mean_rmse = cv_results['test-rmse-mean'].min()
    #boost_rounds = cv_results['test-rmse-mean'].argmin()
    #print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    #if mean_rmse < min_rmse:
    #    min_rmse = mean_rmse
    #    best_params = eta

#print("Best params: {}, RMSE: {}".format(eta, min_rmse))



