#coding=utf-8
"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2021.08.01
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import kurtosis, iqr, skew
import gc
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)

        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_

def parallel_apply(groups, func, index_name='Index', num_workers=1, chunk_size=100000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        for i in features_chunk:
            features.append(i)
    return features

def parallel_apply_fea(groups, func, index_name='Index', num_workers=1, chunk_size=100000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in chunk_groups(groups, chunk_size):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features

def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
        elif agg == 'nunique':
            features['{}{}_nunique'.format(prefix, feature_name)] = gr_[feature_name].nunique()
    return features

def reduce_mem_usage(df):
    # print('reduce_mem_usage_parallel start!')
    # chunk_size = df.columns.shape[0]
    # start_mem = df.memory_usage().sum() / 1024 ** 2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    # end_mem = df.memory_usage().sum() / 1024 ** 2
    # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    # print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def reduce_mem_usage_parallel(df_original,num_worker):
    print('reduce_mem_usage_parallel start!')
    # chunk_size = df_original.columns.shape[0]
    start_mem = df_original.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    if df_original.columns.shape[0]>500:
        group_chunk = []
        for  name in df_original.columns:
            group_chunk.append(df_original[[name]])
        with mp.Pool(num_worker) as executor:
            df_temp = executor.map(reduce_mem_usage,group_chunk)
        del group_chunk
        gc.collect()
        df_original = pd.concat(df_temp,axis = 1)
        end_mem = df_original.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        del df_temp
        gc.collect()
    else:
        df_original = reduce_mem_usage(df_original)
        end_mem = df_original.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df_original

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

def ridge_feature_select(X_train, y_train, num_folds):
    print("Starting feature select. Train shape: {}".format(X_train.shape))
    skf = KFold(n_splits=num_folds, shuffle=True, random_state=2021)
    feature_importance_df = pd.DataFrame()
    oof_preds = np.zeros(X_train.shape[0])

    k_fold_mape = []
    for i, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        clf = Ridge(alpha=1)
        clf.fit(X_train.iloc[trn_idx].fillna(0), y_train.iloc[trn_idx])
        oof_preds[val_idx] = clf.predict(X_train.iloc[val_idx].fillna(0))
        k_fold_mape.append(MAPE(y_train.iloc[val_idx], oof_preds[val_idx]))
        # print("kfold_{}_mape_score:{} ".format(i, k_fold_mape[i]))
    full_mape =  MAPE(y_train, oof_preds)
    print("full_mape_score:{} ".format(full_mape))
    return k_fold_mape,full_mape

def feature_select(X_train,y_train):
    feature_importance_df_ = pd.read_csv('feature_importances.csv')
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    best_features = best_features.groupby('feature',as_index = False)['importance'].mean()
    best_features = best_features.sort_values(by = 'importance',ascending=False)
    data=best_features.sort_values(by="importance", ascending=False)
    feature_select = list(data['feature'].values)
    feature_array = []
    full_mape_all = 0
    count = 0
    for fea in feature_select:
        print(count)
        count = count + 1
        feature_array.append(fea)
        df_select = X_train[feature_array]
        k_fold_mape, full_mape = ridge_feature_select(df_select,y_train, num_folds=5)
        if count == 1:
            full_mape_all = full_mape
            file = open('feature_select_name.txt', 'a')
            file.write(fea + '\n')
            file.close()
            file = open('feature_select_fullauc.txt', 'a')
            file.write(str(full_mape_all) + '\n')
            file.close()
            file = open('feature_select_kfoldauc.txt', 'a')
            file.write(str(k_fold_mape) + '\n')
            file.close()
            del df_select
            gc.collect()
            continue
        if full_mape_all <= full_mape:
            feature_array.remove(fea)
        else:
            full_mape_all = full_mape
            file = open('feature_select_name.txt', 'a')
            file.write(fea + '\n')
            file.close()
            file = open('feature_select_fullauc.txt', 'a')
            file.write(str(full_mape_all) + '\n')
            file.close()
            file = open('feature_select_kfoldauc.txt', 'a')
            file.write(str(k_fold_mape) + '\n')
            file.close()
        del df_select
        gc.collect()
    a = 1

