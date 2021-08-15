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
from utils import reduce_mem_usage,reduce_mem_usage_parallel,lgb_score_mape,MAPE
import gc
import warnings
import os,random,pickle
import optuna
warnings.filterwarnings("ignore")
def slice_id_change(x):
    hour = x * 5 / 60
    hour = np.floor(hour)
    hour += 8
    if hour >= 24:
        hour = hour - 24
    return hour
def optuna_print(tr_x, tr_y, te_x,te_y):
    def objective(trial,tr_x, tr_y, te_x,te_y):
        dtrain = lgb.Dataset(tr_x, label=tr_y)
        dvalid = lgb.Dataset(te_x, label=te_y)
        param = {
            "objective": "regression",
            "metric": "mape",
            "verbosity": -1,
            "boosting_type": "gbdt",
            'min_split_gain': 0,
            'random_state':2021,
            'max_bin':trial.suggest_int('max_bin',63,250),
            'subsample_for_bin': trial.suggest_int('subsample_for_bin', 40000, 300000),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        # Add a callback for pruning.
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "mape")
        gbm = lgb.train(
            param, dtrain, valid_sets=[dvalid], verbose_eval=False, callbacks=[pruning_callback]
        )

        preds = gbm.predict(te_x)
        pred_labels = np.rint(preds)
        mape = MAPE(te_y, pred_labels)
        return mape
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="minimize"
    )
    study.optimize(lambda trial: objective(trial, tr_x, tr_y, te_x, te_y),
                   n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
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
#调参
#tr_x, te_x,tr_y,te_y = train_test_split(X_train,y_train,test_size=0.2,random_state=2021)
#optuna_print(tr_x, tr_y, te_x,te_y)
#del tr_x, te_x,tr_y,te_y
#gc.collect()

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
#use single model feature importance as best_feature_importances
feature_importance_df_ = pd.read_csv('best_feature_importances.csv')
cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
best_features = best_features.groupby('feature',as_index = False)['importance'].mean()
best_features = best_features.sort_values(by = 'importance',ascending=False)
data=best_features.sort_values(by="importance", ascending=False)
feature_select = list(data['feature'].values)
feature_cols = feature_select

random_seed = list(range(2021))
max_depth = [4,4,4,4,5,5,5,5,6,6,6,6,7,7,7]
lambd1 =  np.arange(0, 1, 0.0001)
lambd2 =  np.arange(0, 1, 0.0001)
bagging_fraction = [i / 1000.0 for i in range(700, 800)]
feature_fraction = [i / 1000.0 for i in range(700, 800)]
min_child_weight = [i / 100.0 for i in range(150, 250)]
n_feature = [i / 100.0 for i in range(1, 32,2)]
max_bin = list(range(130, 240))
subsample_for_bin = list(range(50000, 220000,10000))
bagging_freq = [1,2,3,4,5,6,7,8,9,10,1,2,3,4,5]
num_leaves = list(range(130, 250))


random.shuffle(random_seed)
random.shuffle(max_depth)
random.shuffle(lambd1)
random.shuffle(lambd2)
random.shuffle(bagging_fraction)
random.shuffle(feature_fraction)
random.shuffle(min_child_weight)
random.shuffle(max_bin)
random.shuffle(subsample_for_bin)
random.shuffle(bagging_freq)
random.shuffle(num_leaves)
random.shuffle(n_feature)


with open('params.pkl', 'wb') as f:
    pickle.dump((random_seed, max_depth, lambd1,lambd2, bagging_fraction, feature_fraction, min_child_weight, max_bin,subsample_for_bin,bagging_freq,num_leaves,n_feature), f)
for iter in range(15):
    print('max_depth:',max_depth[iter],'random_seed:',random_seed[iter],'feature_fraction:',feature_fraction[iter],
          'bagging_fraction:',bagging_fraction[iter],'min_child_weight:',min_child_weight[iter],
          'lambd1:',lambd1[iter],'lambd2:',lambd2[iter],'max_bin:',max_bin[iter],'num_leaves:',num_leaves[iter]
          ,'subsample_for_bin:',subsample_for_bin[iter],'bagging_freq:',bagging_freq[iter],'n_feature:',n_feature[iter])
nround = 5000
for iter in range(15):
    if max_depth[iter]==4:
        nround = 10000
    elif max_depth[iter]==5:
        nround = 8000
    elif max_depth[iter]==6:
        nround = 6000
    elif max_depth[iter] == 7:
        nround = 5000
    X_train_r = X_train[feature_cols[:int(len(feature_cols)*0.7)]+
             feature_cols[int(len(feature_cols)*0.7):int(len(feature_cols)*0.7)+int(len(feature_cols)*n_feature[iter])]]
    X_test_r = X_test[feature_cols[:int(len(feature_cols) * 0.7)] +
                        feature_cols[int(len(feature_cols) * 0.7):int(len(feature_cols) * 0.7) + int(
                            len(feature_cols) * n_feature[iter])]]
    scores  = 0
    threshold = 0
    print('start training......')
    print('训练集维度：',X_train_r.shape)
    print('测试集维度：',X_test_r.shape)
    for i, (trn_idx, val_idx) in enumerate(skf.split(X_train_r, y_train)):
        clf = lgb.LGBMRegressor(
            boosting_type='gbdt',
            objective='regression',
            n_estimators=nround,
            learning_rate=0.08,
            num_leaves=num_leaves[iter],
            max_bin=max_bin[iter],
            max_depth=max_depth[iter],
            random_state=random_seed[iter],
            subsample_for_bin=subsample_for_bin[iter],
            feature_fraction=feature_fraction[iter],
            bagging_fraction=bagging_fraction[iter],
            bagging_freq=bagging_freq[iter],
            min_child_weight=min_child_weight[iter],
            lambda_l1=lambd1[iter],
            lambda_l2=lambd2[iter],
            metric=None,
            n_jobs=30,
            device='gpu'
        )
        clf.fit(X_train_r.iloc[trn_idx], y_train.iloc[trn_idx], eval_set=[(X_train_r.iloc[trn_idx], y_train.iloc[trn_idx]), (X_train_r.iloc[val_idx], y_train.iloc[val_idx])],eval_metric='mape',verbose=100, early_stopping_rounds=200)

        print('predicting')
        val_predict = clf.predict(X_train_r.iloc[val_idx], num_iteration=clf.best_iteration_)
        test_predict[:,i] = clf.predict(X_test_r, num_iteration=clf.best_iteration_)
        k_fold_mape.append(MAPE(y_train.iloc[val_idx],val_predict))
        print("kfold_{}_mape_score:{} ".format(i, k_fold_mape[i]))

    print('Train set kfold {} mean mape:'.format(i), np.mean(k_fold_mape))
    #display_importances(feature_importance_df)
    test_head['result'] = np.mean(test_predict,axis=1)
    test_head['id'] = test_head['order_id']
    test_head[['id','result']].to_csv('random_result/submission_{}.csv'.format(iter),index=False)
    del X_train_r,X_test_r
    gc.collect()
#merge
count = 0
result = 1
for name in os.listdir('random_result/'):
    tmp = pd.read_csv('random_result/'+name)
    if count == 0:
        result = tmp[['id']]
    tmp = tmp.rename(columns={'result':'result{}'.format(count)})
    result = result.merge(tmp,on='id',how='left')
    count += 1
result['result'] = result.drop('id',axis=1).sum(axis=1)
result['result'] = result['result']/count
result[['id','result']].to_csv('submission_merge.csv',index=False)
