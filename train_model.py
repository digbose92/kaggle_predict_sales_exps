import pandas as pd 
import time 
import numpy as np 
from xgboost import XGBRegressor
import matplotlib.pylab as plt
import pickle 

start_time=time.time()
tot_data=pd.read_csv('../data/tot_train_test_data_corrected_v5.csv')
test_data=pd.read_csv('../data/test.csv').set_index('ID')
elapsed_time=time.time()-start_time
tot_data.fillna(0,inplace=True)
print('Data Loading time:%f'%(elapsed_time))

#remove the first 3 months data 
tot_data = tot_data[tot_data.date_block_num > 3]
print(tot_data.columns)
print(tot_data.shape)
print(tot_data.head())
#tot_data=tot_data.drop('Unnamed: 0',1)
#tot_data=tot_data.drop('item_name',1)


tot_data=tot_data[['date_block_num', 'shop_id', 'item_id', 'target', 'ID',
       'city_id','shop_category', 'item_category_id', 'cat_code',
       'subtype_code', 'target_lag_1', 'target_lag_2', 'target_lag_3','month_avg_item_count_lag_1','month_avg_item_count_lag_2',
       'month_avg_item_count_lag_3','month_avg_item_id_wise_count_lag_1',
       'month_avg_item_id_wise_count_lag_2',
       'month_avg_item_id_wise_count_lag_3', 'month_avg_shop_id_wise_count_lag_1',
       'month_avg_shop_id_wise_count_lag_2',
       'month_avg_shop_id_wise_count_lag_3','month_avg_item_cat_id_wise_count_lag_1','month_avg_item_cat_id_wise_count_lag_2','month_avg_item_cat_id_wise_count_lag_3','month_shop_item_avg_cnt_lag_1', 'month_shop_item_avg_cnt_lag_2',
       'month_shop_item_avg_cnt_lag_3','month_shop_cat_avg_cnt_lag_1','month_shop_cat_avg_cnt_lag_2','month_shop_cat_avg_cnt_lag_3','month_shop_subtype_avg_cnt_lag_1','month_shop_subtype_avg_cnt_lag_2',
       'month_shop_subtype_avg_cnt_lag_3', 'month_city_avg_item_cnt_lag_1', 'month_city_avg_item_cnt_lag_2',
       'month_city_avg_item_cnt_lag_3', 'month_item_wise_city_avg_item_cnt_lag_1', 'month_item_wise_city_avg_item_cnt_lag_2', 'month_item_wise_city_avg_item_cnt_lag_3','month_num', 'days' ]]
#print(tot_data['item_name'])
print(tot_data.shape)
print(tot_data.columns)

X_train = tot_data[tot_data.date_block_num < 33].drop(['target'], axis=1)
Y_train = tot_data[tot_data.date_block_num < 33]['target']
X_valid = tot_data[tot_data.date_block_num == 33].drop(['target'], axis=1)
Y_valid = tot_data[tot_data.date_block_num == 33]['target']
X_test = tot_data[tot_data.date_block_num == 34].drop(['target'], axis=1)
#check if any null data is there 
#print(tot_data.isnull().values.any())

print(X_train.shape,Y_train.shape)
print(X_valid.shape,Y_valid.shape)
print(X_test.shape)


ts = time.time()

model = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    subsample=0.8, 
    eta=0.1,
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 20)

end_time=time.time()

train_time=end_time-ts
print('training time:%f' %(train_time))

### save model
##with open("pima_corrrected_v2.pickle.dat", "wb") as f:
#    pickle.dump(model,f)
#pickle.dump(model, open("pima.pickle.dat", "wb"))

####testing 
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test_data.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('xgb_submission_corrected_new.csv', index=False)