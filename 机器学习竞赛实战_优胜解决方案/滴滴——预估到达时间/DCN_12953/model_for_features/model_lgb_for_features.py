import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import train_test_split


def append_all_data(files_list, file_head_path):
    """
    concat all the data
    :param files_list: the name of data
    :param file_head_path: the path of data
    :return: DataFrame of data for all
    """
    data_all_path = file_head_path + files_list[0]
    data_all = pd.read_csv(data_all_path)
    data_all = data_all.head(0)
    try:
        del data_all['Unnamed: 0']
    except KeyError as e:
        pass
    # 循环添加全部数据
    for i in files_list:
        data_path = file_head_path + i
        print("当前文件为：", data_path)
        data = pd.read_csv(data_path)
        try:
            del data['Unnamed: 0']
        except KeyError as e:
            pass
        data_all = data_all.append(data)
    return data_all


def file_name(file_dir):
    files_list = []
    for root, dirs, files in os.walk(file_dir):
        # print("success")
        for name in files:
            files_list.append(name)
    return files_list


def del_str_in_list(lst, del_str):
    a = []
    for i in range(len(lst)):
        if del_str not in lst[i]:
            a.append(lst[i])
    return a


# 自定义lgb评估指标
def lgb_score_mape(preds, train_data):
    labels = train_data.get_label()
    diff = np.abs(np.array(preds) - np.array(labels))
    result = np.mean(diff / labels)
    return 'mape',result, False

# 评估指标
def MAPE(true, pred):
    diff = np.abs(np.array(pred) - np.array(true))
    return np.mean(diff / true)

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
    mis_val = df.isnull().sum()
        
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
   
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
    #print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
    #    "There are " + str(mis_val_table_ren_columns.shape[0]) +
    #      " columns that have missing values.")
        
       
        # Return the dataframe with missing information
    return mis_val_table_ren_columns


def model_fit(train_x, train_y):
    evals_result = {}
    params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',  # 回归目标
            #'metric': {'binary_logloss,auc'},
            #'max_depth':-1,
            'num_leaves': 30,
            'learning_rate': 0.07,
            #'min_child_samples':21,
            #'min_child_weight':0.001,
            #'feature_fraction': 0.7,
            #'bagging_fraction': 0.6,
            #'bagging_freq': 2,
            #'min_split_gain':0.5,
            'verbose': 0,
            #'is_unbalenced':True,
        }


    n_fold=5
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    # gkf = GroupKFold(n_splits=n_fold)
    toof = np.zeros((train_x.shape[0], ))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x,train_y)):  # 5折训练
        print("fold {}".format(fold_ + 1))
        trn_data = lgb.Dataset(train_x.iloc[trn_idx], label=train_y.iloc[trn_idx])
        val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y.iloc[val_idx])

        clf = lgb.train(params,
                        trn_data,
                        valid_sets=[trn_data, val_data],
                        valid_names=['train', 'val'],
                        verbose_eval=10,
                        feval=lgb_score_mape,
                        #categorical_feature=[],
                        evals_result=evals_result,
                        early_stopping_rounds=20,
                        num_boost_round = 1000
                        )
        toof[val_idx] = clf.predict(train_x.iloc[val_idx], num_iteration=clf.best_iteration)

        #print('拟合情况:')
        #lgb.plot_metric(evals_result)
        #plt.show()
        mape_vale = MAPE(train_y.iloc[val_idx],toof[val_idx])
        print("当前MAPE值为：",mape_vale)

        print('画特征重要性排序...')
        plt.figure(figsize=(10, 30))
        ax = lgb.plot_importance(clf, figsize=(10,30))#max_features表示最多展示出前10个重要性特征，可以自行设置
        plt.savefig("features_importance.png", dpi=500, bbox_inches='tight') 

    return clf


if __name__=='__main__':
    making_data_dir = '/home/didi2021/didi2021/giscup_2021/order_xt/'
    mk_list = file_name(making_data_dir)
    mk_list.sort()
    mk_data = append_all_data(mk_list, making_data_dir)
    print(mk_data.shape)
    mk_data['date_time'] = mk_data['date_time'].astype(int)
    mk_data = mk_data[mk_data['date_time']!=20200901]
    print(mk_data.shape)
    describe_df = mk_data.describe()
    describe_df.to_csv('describe_df.csv')
    print('*-'*40, 'missing_values_table')
    ms_table = missing_values_table(mk_data)
    ms_table.to_csv('missing_values_table.csv')
    train_y = mk_data['ata']
    train_x = mk_data.drop(['ata','weather','date_time_dt','order_id','driver_id','date_time'],axis=1)
    print(train_y)
    print('*-'*40)
    print(train_x.head(5))
    print('*-'*40, 'model_fit')
    model = model_fit(train_x, train_y)
    print('................FINISH')







