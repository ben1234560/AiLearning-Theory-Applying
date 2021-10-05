#coding=utf-8
"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2021.08.01
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import os
import gc
import warnings
from utils import parallel_apply_fea,add_features_in_group
from functools import partial
warnings.filterwarnings("ignore")

def last_k_cross_time_interval(gr, periods):
    gr_ = gr.copy()
    gr_ = gr_.iloc[::-1]
    gr_['t_i_v'] = gr_['cross_time'].diff()
    gr_['t_i_v'] = gr_['t_i_v']
    gr_['t_i_v'] = gr_['t_i_v'].fillna(0)
    gr_ = gr_.drop_duplicates().reset_index(drop = True)

    # cross time变化
    features = {}
    for period in periods:
        if period > 10e5:
            period_name = 'zsl_cross_time_interval_all'
            gr_period = gr_.copy()
        else:
            period_name = 'zsl_cross_time_interval_last_{}_'.format(period)
            gr_period = gr_.iloc[:period]
        features = add_features_in_group(features, gr_period, 't_i_v',
                                             ['mean','max', 'min', 'std','sum'],
                                             period_name)
    return features

# last k cross id time trend
def last_cross_time_features(gr,periods):
    gr_ = gr.copy()
    gr_ = gr_.iloc[::-1]
    features = {}
    for period in periods:
        if period > 10e5:
            period_name = 'zsl_all_'
            gr_period = gr_.copy()
        else:
            period_name = 'zsl_last_{}_'.format(period)
            gr_period = gr_.iloc[:period]
        features = add_features_in_group(features, gr_period, 'cross_time',
                                     ['max', 'sum', 'mean','min','std'],
                                     period_name)
    return features


# last k cross id time trend
def trend_in_last_k_cross_id_time(gr, periods):
    gr_ = gr.copy()
    gr_ = gr_.iloc[::-1]
    features = {}
    for period in periods:
        gr_period = gr_.iloc[:period]
        features = add_trend_feature(features, gr_period,
                                     'cross_time', 'zsl_{}_period_trend_'.format(period)
                                     )
    return features
# trend feature
def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features

def slice_id_change(x):
    hour = x * 5 / 60
    hour = np.floor(hour)
    hour += 8
    if hour >= 24:
        hour = hour - 24
    return hour
if __name__ == '__main__':
    nrows = None
    root_path = '../data/giscup_2021/'
    read_idkey = np.load(root_path + 'id_key_to_connected_allday.npy', allow_pickle=True).item()
    read_grapheb = np.load(root_path + 'graph_embeddings_retp1_directed.npy', allow_pickle=True).item()
    read_grapheb_retp = np.load(root_path + 'graph_embeddings_retp05_directed.npy', allow_pickle=True).item()
    for i in read_grapheb:
        read_grapheb[i] = list(read_grapheb[i]) + list(read_grapheb_retp[i])
    del read_grapheb_retp
    head_columns = ['order_id', 'ata', 'distance', 'simple_eta', 'driver_id','slice_id']
    embedding_k = 256
    fill_list = [0] * embedding_k
    df = []
    #######################################nextlinks #######################################
    nextlinks = pd.read_csv(root_path+'nextlinks.txt', sep=' ', header=None)
    nextlinks.columns=['from_id', 'to_id']
    nextlinks['to_id'] = nextlinks['to_id'].astype('str')
    nextlinks['to_id'] = nextlinks['to_id'].apply(lambda x: x.split(","))
    nextlinks = pd.DataFrame({'from_id':nextlinks.from_id.repeat(nextlinks.to_id.str.len()),
                                        'to_id':np.concatenate(nextlinks.to_id.values)})
    from_id_weight = nextlinks['from_id'].value_counts()
    from_id_weight = from_id_weight.to_frame()
    from_id_weight['index'] = from_id_weight.index

    from_id_weight.columns=['weight', 'from_id']
    nextlinks = pd.merge(nextlinks,from_id_weight, 'left', on=['from_id'])
    nextlinks = nextlinks.sort_values(by='weight',ascending=False)
    G = nx.DiGraph()
    from_id = nextlinks['from_id'].astype(str).to_list()
    to_id = nextlinks['to_id'].to_list()
    weight = nextlinks['weight'].to_list()
    edge_tuple = list(zip(from_id, to_id,weight))
    print('adding')
    G.add_weighted_edges_from(edge_tuple)

    dc = nx.algorithms.centrality.degree_centrality(G)
    dc = sorted(dc.items(), key=lambda d: d[1],reverse=True)
    dc = dc[:50000]
    dc = [str(i[0]) for i in dc ]
    #######################################cross #######################################
    for name in os.listdir(root_path+'train/'):
        data_time = name.split('.')[0]
        if data_time=='20200803':
            continue
        train = pd.read_csv(root_path+'train/{}'.format(name),sep= ';;',header=None,nrows=nrows)
        print("开始处理", data_time)
        train_head = pd.DataFrame(train[0].str.split(' ').tolist(),columns = ['order_id', 'ata', 'distance','simple_eta', 'driver_id', 'slice_id'])
        train_head['order_id'] = train_head['order_id'].astype(str)
        train_head['ata'] = train_head['ata'].astype(float)
        train_head['distance'] = train_head['distance'].astype(float)
        train_head['simple_eta'] = train_head['simple_eta'].astype(float)
        train_head['driver_id'] = train_head['driver_id'].astype(int)
        train_head['slice_id'] = train_head['slice_id'].astype(int)
        # 处理corss数据
        data_cross = train[[2]]
        data_cross['index'] = train_head.index
        data_cross['order_id'] = train_head['order_id']
        data_cross_split = data_cross[2].str.split(' ', expand=True).stack().to_frame()
        data_cross_split = data_cross_split.reset_index(level=1, drop=True).rename(columns={0: 'cross_info'})
        data_cross_split = data_cross[['index', 'order_id']].join(data_cross_split)
        data_cross_split[['cross_id', 'cross_time']] = data_cross_split['cross_info'].str.split(':', 2, expand=True)
        data_cross_split['cross_time'] = data_cross_split['cross_time'].astype(float)
        tmp_cross_id = data_cross_split['cross_id'].str.split('_', expand=True)
        tmp_cross_id.columns=['cross_id_in','cross_id_out']
        data_cross_split = pd.concat([data_cross_split,tmp_cross_id],axis=1).drop(['cross_id','cross_info'],axis=1)
        data_cross_split['date_time'] = data_time
        data_cross_split = data_cross_split.drop('index',axis=1).reset_index(drop=True)
        print('preprocess finish!')
        print('start feature engineering')
        feature = train_head[['order_id', 'distance']]
        ###################static fea#############################################
        data_cross_split['zsl_cross_id_isnull'] =0
        data_cross_split.loc[data_cross_split['cross_id_in'].isnull(),'zsl_cross_id_isnull'] = 1
        data_cross_split.loc[data_cross_split['cross_id_in'].isnull(),'cross_id_in'] = '-1'
        data_cross_split.loc[data_cross_split['cross_id_out'].isnull(),'cross_id_out'] = '-1'
        #######################order cross_id count###############################
        df = data_cross_split.groupby('order_id', as_index=False)
        tmp_crossid_agg = df['cross_id_in'].agg({'zsl_order_cross_id_in_count': 'count'})
        tmp_crossid_agg['zsl_order_cross_id_in_count_bins'] = 0
        tmp_crossid_agg.loc[(tmp_crossid_agg['zsl_order_cross_id_in_count']>=5)&(tmp_crossid_agg['zsl_order_cross_id_in_count']<10),'zsl_order_cross_id_in_count_bins']=1
        tmp_crossid_agg.loc[(tmp_crossid_agg['zsl_order_cross_id_in_count']>=10)&(tmp_crossid_agg['zsl_order_cross_id_in_count']<20),'zsl_order_cross_id_in_count_bins']=2
        tmp_crossid_agg.loc[(tmp_crossid_agg['zsl_order_cross_id_in_count']>=20),'zsl_order_cross_id_in_count_bins']=3
        feature = feature.merge(tmp_crossid_agg,on='order_id',how='left')
        print('order cross_id count finish!')
        #######################order cross id & distance###############################
        feature['zsl_order_cross_is_highspeed'] = 0
        feature.loc[(feature['distance']>90000)&(feature['zsl_order_cross_id_in_count']<30),'zsl_order_cross_is_highspeed'] = 1
        print('order cross id & distance finish!')
        #######################order cross id & nextlinks centry###############################
        tmp = data_cross_split[data_cross_split['cross_id_in'].isin(dc)]
        tmp = tmp.groupby('order_id', as_index=False)
        tmp_linkid_centry_count = tmp['cross_id_in'].agg({'zsl_order_cross_id_in_centry_count': 'count'})
        feature = feature.merge(tmp_linkid_centry_count,on='order_id',how='left')
        feature['zsl_order_cross_id_in_centry_count'] = feature['zsl_order_cross_id_in_centry_count'].fillna(0)
        tmp = data_cross_split[data_cross_split['cross_id_out'].isin(dc)]
        tmp = tmp.groupby('order_id', as_index=False)
        tmp_linkid_centry_count = tmp['cross_id_out'].agg({'zsl_order_cross_id_out_centry_count': 'count'})
        feature = feature.merge(tmp_linkid_centry_count, on='order_id', how='left')
        feature['zsl_order_cross_id_out_centry_count'] = feature['zsl_order_cross_id_out_centry_count'].fillna(0)
        print('order cross_id & nextlinks centry finish!')
        #######################order cross_time sum mean max min var std###############################
        tmp_linktime_agg = df['cross_time'].agg({'zsl_order_cross_time_sum': 'sum','zsl_order_cross_time_mean': 'mean',
                                              'zsl_order_cross_time_max': 'max','zsl_order_cross_time_min': 'min',
                                              'zsl_order_cross_time_var': 'var'})
        feature = feature.merge(tmp_linktime_agg,on='order_id',how='left')
        print('order cross_time sum mean max min var std finish!')
        #######################order distance/link_id_count###############################
        feature['zsl_distance_div_cross_id_count'] = feature['distance']*10/feature['zsl_order_cross_id_in_count']
        feature = feature.drop('distance', axis=1)
        print('order distance div link_id_count finish!')
        ###################trend fea#############################################
        ###################trend cross time#####################################
        groupby = data_cross_split.groupby(['order_id'])
        func = partial(trend_in_last_k_cross_id_time, periods=[2, 5, 10, 20,100000000])
        g = parallel_apply_fea(groupby, func, index_name='order_id', num_workers=5, chunk_size=10000)
        feature = feature.merge(g, on='order_id', how='left')
        func = partial(last_cross_time_features, periods=[2, 5, 10, 20,100000000])
        g = parallel_apply_fea(groupby, func, index_name='order_id', num_workers=5, chunk_size=10000)
        feature = feature.merge(g, on='order_id', how='left')
        func = partial(last_k_cross_time_interval, periods=[2, 5, 10, 20, 100000000])
        g = parallel_apply_fea(groupby, func, index_name='order_id', num_workers=5, chunk_size=10000)
        feature = feature.merge(g, on='order_id', how='left')
        print('trend cross time finish!')
        ####################nextlinks graph embedding#######################
        data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].astype(int)
        data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].map(read_idkey)
        data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].fillna(0)
        data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].astype(int)
        data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].map(read_grapheb)
        data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].fillna('0')
        def replace_list(x):
            if isinstance(x, str):
                x = fill_list
            return x
        data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].apply(replace_list)
        cross_id_in_col = ['zsl_cross_id_in_eb{}'.format(i) for i in range(embedding_k)]
        agg_col = dict(zip(cross_id_in_col, ['mean'] * len(cross_id_in_col)))
        cross_id_in_array = np.array(data_cross_split.pop('cross_id_in').to_list())
        cross_id_in_array = pd.DataFrame(cross_id_in_array, columns=agg_col, dtype=np.float16)
        data_cross_split = pd.concat([data_cross_split, cross_id_in_array], axis=1)
        tmp = data_cross_split.groupby('order_id', as_index=False)
        tmp_crossidin_agg = tmp.agg(agg_col)
        feature = feature.merge(tmp_crossidin_agg, on='order_id', how='left')
        print('trend cross_id_in eb finish!')
        data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].astype(int)
        data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].map(read_idkey)
        data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].fillna(0)
        data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].astype(int)
        data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].map(read_grapheb)
        data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].fillna('0')
        def replace_list(x):
            if isinstance(x, str):
                x = fill_list
            return x
        data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].apply(replace_list)
        cross_id_out_col = ['zsl_cross_id_out_eb{}'.format(i) for i in range(embedding_k)]
        agg_col = dict(zip(cross_id_out_col, ['mean'] * len(cross_id_out_col)))
        cross_id_out_array = np.array(data_cross_split.pop('cross_id_out').to_list())
        cross_id_out_array = pd.DataFrame(cross_id_out_array, columns=agg_col, dtype=np.float16)
        data_cross_split = pd.concat([data_cross_split, cross_id_out_array], axis=1)
        tmp = data_cross_split.groupby('order_id', as_index=False)
        tmp_crossidout_agg = tmp.agg(agg_col)
        feature = feature.merge(tmp_crossidout_agg, on='order_id', how='left')
        print('trend cross_id_out eb finish!')
        multipy_df = []
        multipy_col = []
        for col1, col2 in zip(cross_id_in_col, cross_id_out_col):
            tmp = feature[col1] * feature[col2]
            multipy_df.append(tmp)
            multipy_col.append(col1 + '_mul_' + col2)
        multipy_df = pd.concat(multipy_df, axis=1)
        multipy_df.columns = multipy_col
        feature = pd.concat([feature, multipy_df], axis=1)
        print('trend cross_id_out eb multipy finish!')
        feature.to_csv(root_path + 'feature/train/cross_fea_order_id_level_{}.csv'.format(data_time), index=False)
        del train
        gc.collect()

    test = pd.read_csv(root_path+'20200901_test.txt',sep= ';;',header=None,nrows=nrows)
    test_head = pd.DataFrame(test[0].str.split(' ').tolist(),columns = ['order_id', 'ata', 'distance','simple_eta', 'driver_id', 'slice_id'])
    test_head['order_id'] = test_head['order_id'].astype(str)
    test_head['ata'] = test_head['ata'].astype(float)
    test_head['distance'] = test_head['distance'].astype(float)
    test_head['simple_eta'] = test_head['simple_eta'].astype(float)
    test_head['driver_id'] = test_head['driver_id'].astype(int)
    test_head['slice_id'] = test_head['slice_id'].astype(int)
    # 处理corss数据
    data_cross = test[[2]]
    data_cross['index'] = test_head.index
    data_cross['order_id'] = test_head['order_id']
    data_cross_split = data_cross[2].str.split(' ', expand=True).stack().to_frame()
    data_cross_split = data_cross_split.reset_index(level=1, drop=True).rename(columns={0: 'cross_info'})
    data_cross_split = data_cross[['index', 'order_id']].join(data_cross_split)
    data_cross_split[['cross_id', 'cross_time']] = data_cross_split['cross_info'].str.split(':', 2, expand=True)
    data_cross_split['cross_time'] = data_cross_split['cross_time'].astype(float)
    tmp_cross_id = data_cross_split['cross_id'].str.split('_', expand=True)
    tmp_cross_id.columns = ['cross_id_in', 'cross_id_out']
    data_cross_split = pd.concat([data_cross_split, tmp_cross_id], axis=1).drop(['cross_id', 'cross_info'], axis=1)
    data_cross_split['date_time'] = '20200901'
    data_cross_split = data_cross_split.drop('index', axis=1).reset_index(drop=True)
    print('preprocess finish!')
    print('start feature engineering')
    feature = test_head[['order_id', 'distance']]
    ###################static fea#############################################
    data_cross_split['zsl_cross_id_isnull'] = 0
    data_cross_split.loc[data_cross_split['cross_id_in'].isnull(), 'zsl_cross_id_isnull'] = 1
    data_cross_split.loc[data_cross_split['cross_id_in'].isnull(), 'cross_id_in'] = '-1'
    data_cross_split.loc[data_cross_split['cross_id_out'].isnull(), 'cross_id_out'] = '-1'
    #######################order cross_id count###############################
    df = data_cross_split.groupby('order_id', as_index=False)
    tmp_crossid_agg = df['cross_id_in'].agg({'zsl_order_cross_id_in_count': 'count'})
    tmp_crossid_agg['zsl_order_cross_id_in_count_bins'] = 0
    tmp_crossid_agg.loc[(tmp_crossid_agg['zsl_order_cross_id_in_count'] >= 5) & (
                tmp_crossid_agg['zsl_order_cross_id_in_count'] < 10), 'zsl_order_cross_id_in_count_bins'] = 1
    tmp_crossid_agg.loc[(tmp_crossid_agg['zsl_order_cross_id_in_count'] >= 10) & (
                tmp_crossid_agg['zsl_order_cross_id_in_count'] < 20), 'zsl_order_cross_id_in_count_bins'] = 2
    tmp_crossid_agg.loc[(tmp_crossid_agg['zsl_order_cross_id_in_count'] >= 20), 'zsl_order_cross_id_in_count_bins'] = 3
    feature = feature.merge(tmp_crossid_agg, on='order_id', how='left')
    print('order cross_id count finish!')
    #######################order cross id & distance###############################
    feature['zsl_order_cross_is_highspeed'] = 0
    feature.loc[(feature['distance'] > 90000) & (
                feature['zsl_order_cross_id_in_count'] < 30), 'zsl_order_cross_is_highspeed'] = 1
    print('order cross id & distance finish!')
    #######################order cross id & nextlinks centry###############################
    tmp = data_cross_split[data_cross_split['cross_id_in'].isin(dc)]
    tmp = tmp.groupby('order_id', as_index=False)
    tmp_linkid_centry_count = tmp['cross_id_in'].agg({'zsl_order_cross_id_in_centry_count': 'count'})
    feature = feature.merge(tmp_linkid_centry_count, on='order_id', how='left')
    feature['zsl_order_cross_id_in_centry_count'] = feature['zsl_order_cross_id_in_centry_count'].fillna(0)
    tmp = data_cross_split[data_cross_split['cross_id_out'].isin(dc)]
    tmp = tmp.groupby('order_id', as_index=False)
    tmp_linkid_centry_count = tmp['cross_id_out'].agg({'zsl_order_cross_id_out_centry_count': 'count'})
    feature = feature.merge(tmp_linkid_centry_count, on='order_id', how='left')
    feature['zsl_order_cross_id_out_centry_count'] = feature['zsl_order_cross_id_out_centry_count'].fillna(0)
    print('order cross_id & nextlinks centry finish!')
    #######################order cross_time sum mean max min var std###############################
    tmp_linktime_agg = df['cross_time'].agg({'zsl_order_cross_time_sum': 'sum', 'zsl_order_cross_time_mean': 'mean',
                                             'zsl_order_cross_time_max': 'max', 'zsl_order_cross_time_min': 'min',
                                             'zsl_order_cross_time_var': 'var'})
    feature = feature.merge(tmp_linktime_agg, on='order_id', how='left')
    print('order cross_time sum mean max min var std finish!')
    #######################order distance/link_id_count###############################
    feature['zsl_distance_div_cross_id_count'] = feature['distance'] * 10 / feature['zsl_order_cross_id_in_count']
    feature = feature.drop('distance', axis=1)
    print('order distance div link_id_count finish!')
    ###################trend fea#############################################
    ###################trend cross time#####################################
    groupby = data_cross_split.groupby(['order_id'])
    func = partial(trend_in_last_k_cross_id_time, periods=[2, 5, 10, 20, 100000000])
    g = parallel_apply_fea(groupby, func, index_name='order_id', num_workers=5, chunk_size=10000)
    feature = feature.merge(g, on='order_id', how='left')
    func = partial(last_cross_time_features, periods=[2, 5, 10, 20, 100000000])
    g = parallel_apply_fea(groupby, func, index_name='order_id', num_workers=5, chunk_size=10000)
    feature = feature.merge(g, on='order_id', how='left')
    func = partial(last_k_cross_time_interval, periods=[2, 5, 10, 20, 100000000])
    g = parallel_apply_fea(groupby, func, index_name='order_id', num_workers=5, chunk_size=10000)
    feature = feature.merge(g, on='order_id', how='left')
    print('trend cross time finish!')
    ####################nextlinks graph embedding#######################
    data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].astype(int)
    data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].map(read_idkey)
    data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].fillna(0)
    data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].astype(int)
    data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].map(read_grapheb)
    data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].fillna('0')
    def replace_list(x):
        if isinstance(x, str):
            x = fill_list
        return x
    data_cross_split['cross_id_in'] = data_cross_split['cross_id_in'].apply(replace_list)
    cross_id_in_col = ['zsl_cross_id_in_eb{}'.format(i) for i in range(embedding_k)]
    agg_col = dict(zip(cross_id_in_col, ['mean'] * len(cross_id_in_col)))
    cross_id_in_array = np.array(data_cross_split.pop('cross_id_in').to_list())
    cross_id_in_array = pd.DataFrame(cross_id_in_array, columns=agg_col, dtype=np.float16)
    data_cross_split = pd.concat([data_cross_split, cross_id_in_array], axis=1)
    tmp = data_cross_split.groupby('order_id', as_index=False)
    tmp_crossidin_agg = tmp.agg(agg_col)
    feature = feature.merge(tmp_crossidin_agg, on='order_id', how='left')
    print('trend cross_id_in eb finish!')
    data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].astype(int)
    data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].map(read_idkey)
    data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].fillna(0)
    data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].astype(int)
    data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].map(read_grapheb)
    data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].fillna('0')
    def replace_list(x):
        if isinstance(x, str):
            x = fill_list
        return x
    data_cross_split['cross_id_out'] = data_cross_split['cross_id_out'].apply(replace_list)
    cross_id_out_col = ['zsl_cross_id_out_eb{}'.format(i) for i in range(embedding_k)]
    agg_col = dict(zip(cross_id_out_col, ['mean'] * len(cross_id_out_col)))
    cross_id_out_array = np.array(data_cross_split.pop('cross_id_out').to_list())
    cross_id_out_array = pd.DataFrame(cross_id_out_array, columns=agg_col, dtype=np.float16)
    data_cross_split = pd.concat([data_cross_split, cross_id_out_array], axis=1)
    tmp = data_cross_split.groupby('order_id', as_index=False)
    tmp_crossidout_agg = tmp.agg(agg_col)
    feature = feature.merge(tmp_crossidout_agg, on='order_id', how='left')
    print('trend cross_id_out eb finish!')
    multipy_df = []
    multipy_col = []
    for col1, col2 in zip(cross_id_in_col, cross_id_out_col):
        tmp = feature[col1] * feature[col2]
        multipy_df.append(tmp)
        multipy_col.append(col1 + '_mul_' + col2)
    multipy_df = pd.concat(multipy_df, axis=1)
    multipy_df.columns = multipy_col
    feature = pd.concat([feature, multipy_df], axis=1)
    print('trend cross_id_out eb multipy finish!')
    feature.to_csv(root_path + 'feature/test/cross_fea_order_id_level_20200901.csv', index=False)
