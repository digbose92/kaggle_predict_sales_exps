import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime 
from collections import Counter
from sklearn import preprocessing
import re 
from itertools import product
import time 
#column information
def remove_outliers(train_data):
    print('Minimum price in training data:',train_data['item_price'].min())#minimum price value
    print('Maximum price in training data:',train_data['item_price'].max()) #maximum price value 
    
    #remove the outliers from the data 
    train_data=train_data[train_data.item_price!=-1.0] #removing the minimum price rows (price = -1.0) (update to all negative values)
    train_data=train_data[train_data.item_price!=307980.0] #removing the maximum price rows (price = 307980)
    train_data=train_data[train_data.item_cnt_day>0]
    #drop duplicates from train_data
    subset = ['date','date_block_num','shop_id','item_id','item_cnt_day']
    train_data.drop_duplicates(subset=subset, inplace=True)
    #train_data is now cleaned data (outliers removed)
    #check the unqiue number of shops (total length of shop data is 60)
    
    return(train_data)

def merge_shop_info(train_data,test_data):
    """ Merging the data from different shops which are same together """
    ##### train data merging 
    #combine the data here (0 and 57 together)
    train_data.loc[train_data.shop_id == 0, 'shop_id'] = 57
    #combine the data here (1 and 58 together)
    train_data.loc[train_data.shop_id == 1, 'shop_id'] = 58
    #combine the data here ( 11 and 10 together)
    train_data.loc[train_data.shop_id == 11, 'shop_id'] = 10
    #combine the data here (40 and 39 together)
    train_data.loc[train_data.shop_id == 40, 'shop_id'] = 39

    #### test data merging 
    #combine the data here (0 and 57 together)
    test_data.loc[test_data.shop_id == 0, 'shop_id'] = 57
    #combine the data here (1 and 58 together)
    test_data.loc[test_data.shop_id == 1, 'shop_id'] = 58
    #combine the data here ( 11 and 10 together)
    test_data.loc[test_data.shop_id == 11, 'shop_id'] = 10
    #combine the data here (40 and 39 together)
    test_data.loc[test_data.shop_id == 40, 'shop_id'] = 39

    return(train_data,test_data)


def generate_shop_data(shops_data):
    """ Check the shops data and group the shops by categories """
    shops_data.loc[shops_data.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"' #same name (just separated by space)
    shop_list=list(shops_data.shop_name)
    city_names=[shop.split(' ')[0] for shop in shop_list]
    category_names=[shop.split(' ')[1] for shop in shop_list]
    shops_data['category']=category_names
    shops_data['city']=city_names 
    xc=shops_data.groupby(['category']).sum()
    xc = xc.sort_values(by ='shop_id',ascending=False)

    #find the index values greater than 100
    xc=xc[xc.shop_id > 100]
    index_list=list(xc.index)
    shops_data['category'] = shops_data['category'].apply(lambda x: x if (x in index_list) else 'etc')
    print('Category Distribution', shops_data.groupby(['category']).sum())
    le_shops=preprocessing.LabelEncoder()

    #do label encoding for the shops data 
    #shops will have the columns shop_city, shop_category and shop_id
    shops_data['city_id']=le_shops.fit_transform(shops_data['city'])
    shops_data['category_id']=le_shops.fit_transform(shops_data['category'])
    shops_data = shops_data[['shop_id','city_id', 'category_id']]

    return(shops_data)

def process_category_data(category_data):
    category_name=category_data['item_category_name']
    category_item_name=[cat.split(' ')[0] for cat in category_name]
    category_data['cat_code']=category_item_name
    count_category_item=Counter(category_item_name)
    #find all the category names greater than 2
    #if greater than 2, then 
    #print(Counter(category_item_name))
    count_stat=Counter(el for el in count_category_item.elements() if count_category_item[el] > 2)
    category_selected=list(count_stat.keys())
    category_data['cat_code']=category_data['cat_code'].apply(lambda x: x if (x in category_selected) else 'etc')
    le_category=preprocessing.LabelEncoder()
    category_data['cat_code'] = le_category.fit_transform(category_data['cat_code'])
    #print(category_data['cat_code'])
    le_subtype_category=preprocessing.LabelEncoder()
    category_data['split'] = category_data['item_category_name'].apply(lambda x: x.split('-'))
    category_data['subtype_code'] = category_data['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    category_data['subtype_code'] = le_subtype_category.fit_transform(category_data['subtype_code'])
    category_data = category_data[['item_category_id','cat_code', 'subtype_code']]
    return(category_data)


def lag_feature_generate(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df


def generate_train_test_pairs(train_data,test_data,items_data,shops_data,category_data):
    matrix = []
    cols = ['date_block_num','shop_id','item_id']
    for block_num in list(train_data['date_block_num'].unique()):
        shop_curr=train_data.loc[train_data['date_block_num'] == block_num, 'shop_id'].unique()
        item_curr=train_data.loc[train_data['date_block_num'] == block_num, 'item_id'].unique()
        matrix.append(np.array(list(product([block_num], train_data.shop_id.unique(), train_data.item_id.unique())), dtype='int16')) #product here gives the cartesian product of shop id and item id for that month

    stack_data = pd.DataFrame(np.vstack(matrix), columns = cols,dtype=np.int32)
    group_data=train_data.groupby(cols,as_index=False)['item_cnt_day'].agg({'target':'sum'}) #aggregates for each month for each (shop,item) pair the number of items sold
    tot_data = pd.merge(stack_data, group_data, how='left', on=cols).fillna(0)
    tot_data['target']=tot_data['target'].clip(0,20) #clipping the target value to (0,20) (same range as )

    #merge test data with train data here 
    test_data['date_block_num']=34
    tot_data=pd.concat([tot_data,test_data],ignore_index=True, sort=False, keys=cols)
    tot_data.fillna(0,inplace=True)


    #merge with items data
    tot_data=pd.merge(tot_data,items_data,on=['item_id'],how='left')
    #merge with shops data
    tot_data=pd.merge(tot_data,shops_data,on=['shop_id'],how='left')
    #merge with category data
    tot_data=pd.merge(tot_data,category_data,on=['item_category_id'],how='left')

    del(group_data)

    #mean number of items sold for every month
    group_data=tot_data.groupby(['date_block_num'],as_index=False)['target'].agg({'month_avg_item_count':'mean'})
    tot_data=pd.merge(tot_data,group_data,on=['date_block_num'],how='left')
    tot_data=lag_feature_generate(tot_data,[1],'month_avg_item_count')
    tot_data.drop(['month_avg_item_count'],axis=1,inplace=True)

    #mean and sum for each item and in each month
    item_cols=['date_block_num','item_id']
    group_data=tot_data.groupby(item_cols,as_index=False)['target'].agg({'month_avg_item_id_wise_count':'mean','month_total_item_id_wise_count':'sum'})
    tot_data = pd.merge(tot_data, group_data, how='left', on=item_cols).fillna(0)
    tot_data=lag_feature_generate(tot_data, [1,2,3], 'month_avg_item_id_wise_count')
    tot_data.drop(['month_avg_item_id_wise_count'],axis=1,inplace=True)
    del(group_data)

    #mean and sum for each shop and in each month
    shop_cols=['date_block_num','shop_id']
    group_data=tot_data.groupby(shop_cols,as_index=False)['target'].agg({'month_avg_shop_id_wise_count':'mean','month_total_shop_id_wise_count':'sum'})
    tot_data = pd.merge(tot_data, group_data, how='left', on=shop_cols).fillna(0)
    tot_data=lag_feature_generate(tot_data,[1,2,3],'month_avg_shop_id_wise_count')
    tot_data.drop(['month_avg_shop_id_wise_count'],axis=1,inplace=True)
    del group_data

    #mean and sum for each item category and in each month
    item_category_cols=['date_block_num','item_category_id']
    group_data=tot_data.groupby(item_category_cols,as_index=False)['target'].agg({'month_avg_item_cat_id_wise_count':'mean','month_total_item_cat_id_wise_count':'sum'})
    tot_data = pd.merge(tot_data, group_data, how='left', on=item_category_cols).fillna(0)
    tot_data=lag_feature_generate(tot_data,[1],'month_avg_item_cat_id_wise_count')
    tot_data.drop(['month_avg_item_cat_id_wise_count'],axis=1,inplace=True)
    del group_data

    #mean for each item id and shop id and date_block_num(month)
    cols_select=['date_block_num','item_id','shop_id']
    group_data=tot_data.groupby(cols_select,as_index=False)['target'].agg({'month_shop_item_avg_cnt':'mean'})
    tot_data = pd.merge(tot_data, group_data, how='left', on=cols_select).fillna(0)
    tot_data=lag_feature_generate(tot_data,[1],'month_shop_item_avg_cnt')
    tot_data.drop(['month_shop_item_avg_cnt'],axis=1,inplace=True)
    del group_data

    #mean and sum for each item category id and shop id and date_block_num(month)
    cols_select=['date_block_num','item_category_id','shop_id']
    group_data=tot_data.groupby(cols_select,as_index=False)['target'].agg({'month_shop_cat_avg_cnt':'mean','month_shop_total_cnt':'sum'})
    tot_data = pd.merge(tot_data, group_data, how='left', on=cols_select).fillna(0)
    tot_data=lag_feature_generate(tot_data,[1],'month_shop_cat_avg_cnt')
    tot_data.drop(['month_shop_cat_avg_cnt'],axis=1,inplace=True)
    del group_data

    #mean and sum for each shop id and subtype code per month
    cols_select=['date_block_num','shop_id','subtype_code']
    group_data=tot_data.groupby(cols_select,as_index=False)['target'].agg({'month_shop_subtype_avg_cnt':'mean','month_shop_subtype_total_cnt':'sum'})
    tot_data = pd.merge(tot_data, group_data, how='left', on=cols_select).fillna(0)
    tot_data=lag_feature_generate(tot_data,[1],'month_shop_subtype_avg_cnt')
    tot_data.drop(['month_shop_subtype_avg_cnt'],axis=1,inplace=True)
    del group_data

    #mean and sum for each shop city per month
    cols_select=['date_block_num','city_id']
    group_data=tot_data.groupby(cols_select,as_index=False)['target'].agg({'month_city_avg_item_cnt':'mean','month_city_total_item_cnt':'sum'})
    tot_data = pd.merge(tot_data, group_data, how='left', on=cols_select).fillna(0)
    tot_data=lag_feature_generate(tot_data,[1],'month_city_avg_item_cnt')
    tot_data.drop(['month_city_avg_item_cnt'],axis=1,inplace=True)
    del group_data

    cols_select=['date_block_num','item_id','city_id']
    group_data=tot_data.groupby(cols_select,as_index=False)['target'].agg({'month_item_wise_city_avg_item_cnt':'mean','month_item_wise_city_total_item_cnt':'sum'})
    tot_data = pd.merge(tot_data, group_data, how='left', on=cols_select).fillna(0)
    tot_data=lag_feature_generate(tot_data,[1],'month_item_wise_city_avg_item_cnt')
    tot_data.drop(['month_item_wise_city_avg_item_cnt'],axis=1,inplace=True)
    del group_data


    #months and date information
    tot_data['month_num']=tot_data['date_block_num'] % 12 
    days_list= pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
    tot_data['days']=tot_data['month_num'].map(days_list)

    return(tot_data)
    
#def generate_train_data(tot_data):



if __name__ == "__main__":
   
    train_data=pd.read_csv('../data/sales_train.csv')
    test_data=pd.read_csv('../data/test.csv')
    items_data=pd.read_csv('../data/items.csv')
    category_data=pd.read_csv('../data/item_categories.csv')
    shops_data=pd.read_csv('../data/shops.csv')

    #removing outliers from training data
    print('===== REMOVING OUTLIERS FROM DATA =====')
    train_data=remove_outliers(train_data)
    print('===== MERGING SHOP INFORMATION =====')
    #merge shop information
    train_data,test_data=merge_shop_info(train_data,test_data)

    print(train_data.shape)
    print(test_data.shape)

    print("===== SHOP CITY CODE GENERATION =====")
    shops_data=generate_shop_data(shops_data)

    print("===== CATEGORY DATA PREPROCESSING =====")
    category_data=process_category_data(category_data)

    print("===== GENERATE THE TRAIN AND TEST PAIRS COMBINED =====")
    start_time=time.time()
    tot_data=generate_train_test_pairs(train_data,test_data,items_data,shops_data,category_data)
    end_time=time.time()
    elapsed_time=end_time-start_time
    print('Total time elapsed:%f' %(elapsed_time))
    #print("===== GENERATE THE TRAIN DATA =====")
    #tot_data=generate_train_d(tot_data,items_data,shops_data,category_data)
    print(tot_data.shape)
    print(tot_data.columns)

    tot_data.to_csv('../data/tot_train_test_data.csv')

    







