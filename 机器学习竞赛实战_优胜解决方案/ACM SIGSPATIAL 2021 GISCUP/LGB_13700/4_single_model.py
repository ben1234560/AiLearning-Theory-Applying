#coding=utf-8
"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2021.08.01
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import lightgbm as lgb
from utils import reduce_mem_usage,reduce_mem_usage_parallel
import os
import gc
import warnings
import time
warnings.filterwarnings("ignore")
def slice_id_change(x):
    hour = x * 5 / 60
    hour = np.floor(hour)
    hour += 8
    if hour >= 24:
        hour = hour - 24
    return hour
# 评估指标
def MAPE(true, pred):
    diff = np.abs(np.array(pred) - np.array(true))
    return np.mean(diff / true)
# 自定义lgb评估指标
def lgb_score_mape(train_data,preds):
    labels = train_data
    diff = np.abs(np.array(preds) - np.array(labels))
    result = np.mean(diff / labels)
    return 'mape',result, False
head_columns = ['order_id', 'ata', 'distance', 'simple_eta', 'driver_id','slice_id']
result = []
result_time_weight = []
result_dis_weight = []
count = 0
df = []
nrows=None
root_path = '../data/giscup_2021/'
data_list = ['20200818', '20200819', '20200820', '20200821', '20200822', '20200823', '20200824',
             '20200825', '20200826', '20200827', '20200828', '20200829', '20200830', '20200831']
#######################################本地验证#######################################
for name in os.listdir(root_path+'train/'):
    data_time = name.split('.')[0]
    if data_time not in data_list:
        continue
    train = pd.read_csv(root_path+'train/{}'.format(name),sep= ';;',header=None,nrows=nrows)
    feature_cross = pd.read_csv(root_path+'feature/train/cross_fea_order_id_level_{}.csv'.format(data_time),nrows=nrows)
    feature_link = pd.read_csv(root_path+'feature/train/link_fea_order_id_level_{}.csv'.format(data_time),nrows=nrows)
    feature_head = pd.read_csv(root_path+'feature/train/head_link_{}.csv'.format(data_time),nrows=nrows)
    feature_sqe = pd.read_csv(root_path + 'feature/train/{}.csv'.format(data_time),nrows=nrows)


    feature_cross['order_id'] = feature_cross['order_id'].astype(str)
    feature_link['order_id'] = feature_link['order_id'].astype(str)
    feature_head['order_id'] = feature_head['order_id'].astype(str)
    feature_sqe['order_id'] = feature_sqe['order_id'].astype(str)

    print("开始处理", data_time)
    # train.columns = ['head','link','cross']
    # train['head'] = train['head'].apply(lambda x:x.split(' '))
    train_head = pd.DataFrame(train[0].str.split(' ').tolist(),columns = ['order_id', 'ata', 'distance','simple_eta', 'driver_id', 'slice_id'])
    train_head['order_id'] = train_head['order_id'].astype(str)
    train_head['ata'] = train_head['ata'].astype(float)
    train_head['distance'] = train_head['distance'].astype(float)
    train_head['simple_eta'] = train_head['simple_eta'].astype(float)
    train_head['driver_id'] = train_head['driver_id'].astype(int)
    train_head['slice_id'] = train_head['slice_id'].astype(int)
    train_head['date_time'] = int(data_time)

    train_head = train_head.merge(feature_cross,on='order_id',how='left')
    train_head = train_head.merge(feature_link,on='order_id',how='left')

    feature_head = feature_head.drop(['ata', 'distance', 'simple_eta', 'driver_id', 'slice_id', 'index',
                                      'date_time', 'link_count', 'link_time_sum', 'link_ratio_sum',
                                      'date_time_dt', 'weekday', 'hour', 'weather', 'hightemp', 'lowtemp',
                                      'len_tmp',
                                      'link_time_mean', 'link_time_std'],
                                     axis=1)
    feature_sqe = feature_sqe.drop(['pre_arrival_status', 'arrive_slice_id', 'slice_id'], axis=1)
    train_head = train_head.merge(feature_sqe, on='order_id', how='left')
    train_head = train_head.merge(feature_head, on='order_id', how='left')

    print('merge finish!')
    train_head = reduce_mem_usage_parallel(train_head,28)
    df.append(train_head.drop('order_id',axis=1))
    del train
    gc.collect()
    count +=1
df = pd.concat(df,axis=0)

test = pd.read_csv(root_path+'20200901_test.txt',sep= ';;',header=None,nrows=nrows)
test_head = pd.DataFrame(test[0].str.split(' ').tolist(),columns = ['order_id', 'ata', 'distance','simple_eta', 'driver_id', 'slice_id'])
test_head['order_id'] = test_head['order_id'].astype(str)
test_head['ata'] = test_head['ata'].astype(float)
test_head['distance'] = test_head['distance'].astype(float)
test_head['simple_eta'] = test_head['simple_eta'].astype(float)
test_head['driver_id'] = test_head['driver_id'].astype(int)
test_head['slice_id'] = test_head['slice_id'].astype(int)


feature_cross = pd.read_csv(root_path + 'feature/test/cross_fea_order_id_level_{}.csv'.format('20200901'),nrows=nrows)
feature_link = pd.read_csv(root_path + 'feature/test/link_fea_order_id_level_{}.csv'.format('20200901'), nrows=nrows)
feature_head = pd.read_csv(root_path + 'feature/test/head_link_{}.csv'.format('20200901'),nrows=nrows)
feature_sqe = pd.read_csv(root_path + 'feature/test/{}.csv'.format('20200901'),nrows=nrows)

test_head['date_time'] = 20200901

feature_cross['order_id'] = feature_cross['order_id'].astype(str)
feature_link['order_id'] = feature_link['order_id'].astype(str)
feature_head['order_id'] = feature_head['order_id'].astype(str)
feature_sqe['order_id'] = feature_sqe['order_id'].astype(str)

test_head = test_head.merge(feature_cross, on='order_id', how='left')

test_head = test_head.merge(feature_link,on='order_id',how='left')

feature_head = feature_head.drop(['ata', 'distance', 'simple_eta', 'driver_id', 'slice_id', 'index',
                                  'date_time', 'link_count', 'link_time_sum', 'link_ratio_sum',
                                  'date_time_dt', 'weekday', 'hour', 'weather', 'hightemp', 'lowtemp',
                                  'len_tmp',
                                  'link_time_mean', 'link_time_std'],
                                 axis=1)
feature_sqe = feature_sqe.drop(['pre_arrival_status', 'arrive_slice_id', 'slice_id'], axis=1)
test_head = test_head.merge(feature_sqe, on='order_id', how='left')
test_head = test_head.merge(feature_head, on='order_id', how='left')

test_head = reduce_mem_usage_parallel(test_head,28)
del feature_cross,feature_link
gc.collect()

X_train = df.drop('ata',axis=1)
y_train = df['ata']
X_test = test_head.drop(['order_id','ata'],axis=1)

folds = 5
skf = KFold(n_splits=folds, shuffle=True, random_state=2021)
train_mean = np.zeros(shape=[1,folds])
test_predict = np.zeros(shape=[X_test.shape[0], folds],dtype=float)
k_fold_mape = []
feature_importance_df = pd.DataFrame()
# Display/plot feature importance
def display_importances(feature_importance_df_):
    feature_importance_df_.to_csv('feature_importances.csv',index=False)
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:100].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    best_features = best_features.groupby('feature',as_index = False)['importance'].mean()
    best_features = best_features.sort_values(by = 'importance',ascending=False)
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('feature_importances.jpg')
    # plt.show()

scores  = 0
threshold = 0
print('start training......')
print('训练集维度：',X_train.shape)
print('测试集维度：',X_test.shape)
for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    clf = lgb.LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        n_estimators=10000,
        learning_rate=0.1,
        num_leaves=170,
        max_bin=63,
        max_depth=-1,
        random_state = 2021,
        subsample_for_bin=200000,
        feature_fraction=0.84,
        bagging_fraction=0.86,
        bagging_freq=7,
        min_child_samples=89,
        lambda_l1=0.006237830242067111,
        lambda_l2=2.016472023736186e-05,
        metric=None,
        n_jobs = 30,
      #  device='gpu'
    )
    clf.fit(X_train.iloc[trn_idx], y_train.iloc[trn_idx], eval_set=[(X_train.iloc[trn_idx], y_train.iloc[trn_idx])
        , (X_train.iloc[val_idx], y_train.iloc[val_idx])],
            eval_metric=lambda y_true, y_pred:[lgb_score_mape(y_true, y_pred)],
     verbose=100, early_stopping_rounds=100)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = X_train.columns
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('predicting')
    val_predict = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration_)
    test_predict[:,i] = clf.predict(X_test, num_iteration=clf.best_iteration_)

    k_fold_mape.append(MAPE(y_train.iloc[val_idx],val_predict))
    print("kfold_{}_mape_score:{} ".format(i, k_fold_mape[i]))

print('Train set kfold {} mean mape:'.format(i), np.mean(k_fold_mape))
display_importances(feature_importance_df)
test_head['result'] = np.mean(test_predict,axis=1)
test_head['id'] = test_head['order_id']
test_head[['id','result']].to_csv('submission.csv',index=False)
