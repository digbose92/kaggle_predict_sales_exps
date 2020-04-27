#random forest regression
import os
import gc
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_pickle('../data/tot_data_new_v2.pkl')
test  = pd.read_csv('../data/test.csv').set_index('ID')
data.fillna(0,inplace=True)

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

"""print('Training the first random forest regression model')
regr_1 = RandomForestRegressor(n_estimators=500,max_depth=20,min_samples_split=5,min_samples_leaf=4,bootstrap=True,max_features="sqrt",verbose=1,n_jobs=45)
regr_1.fit(X_train,Y_train)
valid_predict_1=regr_1.predict(X_valid).clip(0, 20)

mse_1=mean_squared_error(valid_predict_1,Y_valid)
print('Validation RMSE1:%f:'%(sqrt(mse_1)))

#predict using regr_1 on X_test
Y_test_1 = regr_1.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
   "item_cnt_month": Y_test_1
})
submission.to_csv('rf_submission_1.csv', index=False)"""

print('Training the second random forest regression model')
regr_3 = RandomForestRegressor(n_estimators=500,max_depth=30,min_samples_split=5,min_samples_leaf=4,bootstrap=True,max_features="sqrt",verbose=1,n_jobs=45)
regr_3.fit(X_train,Y_train)
valid_predict_3=regr_3.predict(X_valid).clip(0, 20)

mse_3=mean_squared_error(valid_predict_3,Y_valid)
print('Validation RMSE2:%f:'%(sqrt(mse_3)))

#predict using regr_2 on X_test
Y_test_3 = regr_3.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
   "item_cnt_month": Y_test_3
})
submission.to_csv('rf_submission_3.csv', index=False)