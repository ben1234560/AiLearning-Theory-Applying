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

def last_k_link_time_interval(gr, periods):
    gr_ = gr.copy()
    gr_ = gr_.iloc[::-1]
    gr_['t_i_v'] = gr_['link_time'].diff()
    gr_['t_i_v'] = gr_['t_i_v']
    gr_['t_i_v'] = gr_['t_i_v'].fillna(0)

    gr_['c_s_v'] = gr_['link_current_status'].diff()
    gr_['c_s_v'] = gr_['c_s_v']
    gr_['c_s_v'] = gr_['c_s_v'].fillna(0)

    gr_ = gr_.drop_duplicates().reset_index(drop = True)

    # link time变化
    features = {}
    for period in periods:
        if period > 10e5:
            period_name = 'zsl_link_time_interval_all'
            gr_period = gr_.copy()
        else:
            period_name = 'zsl_link_time_interval_last_{}_'.format(period)
            gr_period = gr_.iloc[:period]
        features = add_features_in_group(features, gr_period, 't_i_v',
                                             ['mean','max', 'min', 'std','skew','sum'],
                                             # ['diff'],
                                             period_name)
    # current status变化
    for period in periods:
        if period > 10e5:
            period_name = 'zsl_link_current_status_interval_all'
            gr_period = gr_.copy()
        else:
            period_name = 'zsl_link_current_status_interval_last_{}_'.format(period)
            gr_period = gr_.iloc[:period]
        features = add_features_in_group(features, gr_period, 'c_s_v',
                                     ['mean', 'std', 'skew'],
                                     # ['diff'],
                                     period_name)
    return features

# last k link id time trend
def last_link_time_features(gr,periods):
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
        features = add_features_in_group(features, gr_period, 'link_time',
                                     ['max', 'sum', 'mean','min','skew','std'],
                                     period_name)
        features = add_features_in_group(features, gr_period, 'link_current_status',
                                         ['mean', 'nunique'],
                                         period_name)
    return features
# last k link id time trend
def trend_in_last_k_link_id_time(gr, periods):
    gr_ = gr.copy()
    gr_ = gr_.iloc[::-1]
    features = {}
    for period in periods:
        gr_period = gr_.iloc[:period]
        features = add_trend_feature(features, gr_period,
                                     'link_time', 'zsl_{}_period_trend_'.format(period)
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
    #######################################link #######################################
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
        #link preprocess
        data_link = train[[1]]
        data_link['index'] = train_head.index
        data_link['order_id'] = train_head['order_id']
        data_link['ata'] = train_head['ata']
        data_link['distance'] = train_head['distance']
        data_link['simple_eta'] = train_head['simple_eta']
        data_link['slice_id'] = train_head['slice_id']

        # data_link['slice_id'] = data_link['slice_id'].apply(slice_id_change)
        gc.collect()
        data_link_split = data_link[1].str.split(' ', expand=True).stack().to_frame()
        data_link_split = data_link_split.reset_index(level=1, drop=True).rename(columns={0: 'link_info'})
        # data_link_split = data_link_split.reset_index(drop=True)
        data_link_split = data_link[['order_id', 'index', 'ata', 'distance', 'simple_eta', 'slice_id']].join(
            data_link_split)
        data_link_split = data_link_split.reset_index(drop=True)
        data_link_split[['link_id',
                         'link_time',
                         'link_ratio',
                         'link_current_status',
                         'link_arrival_status']] = data_link_split['link_info'].str.split(':|,', 5, expand=True)
        data_link_split = data_link_split.drop(['link_info'], axis=1)
        data_link_split['link_ratio'] = data_link_split['link_ratio'].astype(float)
        data_link_split['link_time'] = data_link_split['link_time'].astype(float)
        data_link_split['link_current_status'] = data_link_split['link_current_status'].astype(int)
        print('preprocess finish!')
        print('start feature engineering')
        feature = train_head[['order_id', 'distance']]
        ###################static fea#############################################
        #######################order link id count###############################
        df = data_link_split.groupby('order_id', as_index=False)
        tmp_linkid_agg = df['link_id'].agg({'zsl_order_link_id_count': 'count'})
        tmp_linkid_agg['zsl_order_link_id_count_bins'] = 0
        tmp_linkid_agg.loc[(tmp_linkid_agg['zsl_order_link_id_count']>=75)&(tmp_linkid_agg['zsl_order_link_id_count']<100),'zsl_order_link_id_count_bins']=1
        tmp_linkid_agg.loc[(tmp_linkid_agg['zsl_order_link_id_count']>=100)&(tmp_linkid_agg['zsl_order_link_id_count']<120),'zsl_order_link_id_count_bins']=2
        tmp_linkid_agg.loc[(tmp_linkid_agg['zsl_order_link_id_count']>=120),'zsl_order_link_id_count_bins']=3
        feature = feature.merge(tmp_linkid_agg,on='order_id',how='left')
        print('order link id count finish!')
        #######################order link id & distance###############################
        feature['zsl_order_is_highspeed'] = 0
        feature.loc[(feature['distance']>90000)&(feature['zsl_order_link_id_count']<300),'zsl_order_is_highspeed'] = 1
        print('order link id & distance finish!')
        #######################order link id & nextlinks centry###############################
        tmp = data_link_split[data_link_split['link_id'].isin(dc)]
        tmp = tmp.groupby('order_id', as_index=False)
        tmp_linkid_centry_count = tmp['link_id'].agg({'zsl_order_link_id_centry_count': 'count'})
        feature = feature.merge(tmp_linkid_centry_count,on='order_id',how='left')
        feature['zsl_order_link_id_centry_count'] = feature['zsl_order_link_id_centry_count'].fillna(0)
        print('order link id & nextlinks centry finish!')
        #######################order link time sum mean max min var std###############################
        tmp_linktime_agg = df['link_time'].agg({'zsl_order_link_time_sum': 'sum','zsl_order_link_time_mean': 'mean',
                                              'zsl_order_link_time_max': 'max','zsl_order_link_time_min': 'min',
                                              'zsl_order_link_time_var': 'var','zsl_order_link_time_skew': 'skew'})
        feature = feature.merge(tmp_linktime_agg,on='order_id',how='left')
        print('order link time sum mean max min var std finish!')
        #######################order link current status mean nunique###############################
        tmp_linktime_agg = df['link_current_status'].agg({'zsl_link_current_status_mean': 'mean', 'zsl_link_current_status_nunique': 'nunique'})
        feature = feature.merge(tmp_linktime_agg, on='order_id', how='left')
        print('order link current status mean nunique finish!')
        #######################order link current status count vector###############################
        data_link_split['link_current_status'] = data_link_split['link_current_status'].astype(str)
        data_link_split.loc[data_link_split['link_current_status'].astype(int)<0,'link_current_status'] = '0'
        data_link_split.loc[data_link_split['link_current_status'].astype(int)>3,'link_current_status'] = '3'
        data = data_link_split.groupby('order_id')['link_current_status'].apply(lambda x: x.str.cat(sep=',')).reset_index()
        cv_encode = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
        train_x = cv_encode.fit_transform(data['link_current_status'])
        train_x = train_x.toarray()
        link_current_status = pd.DataFrame(train_x, columns=['zsl_link_current_status0', 'zsl_link_current_status1', 'zsl_link_current_status2',
                                           'zsl_link_current_status3'])
        data = pd.concat([data[['order_id']],link_current_status],axis=1)
        feature = feature.merge(data, on='order_id', how='left')
        print('order link current status count vector finish!')
        #######################order distance/link_id_count###############################
        feature['zsl_distance_div_link_id_count'] = feature['distance']*10/feature['zsl_order_link_id_count']
        feature = feature.drop('distance', axis=1)
        print('order distance div link_id_count finish!')
        #######################order link ratio sum mean max min var std###############################
        tmp_linkratio_agg = df['link_ratio'].agg({'zsl_order_link_ratio_sum': 'sum', 'zsl_order_link_ratio_mean': 'mean',
                                                 'zsl_order_link_ratio_min': 'min',
                                                'zsl_order_link_ratio_var': 'var', 'zsl_order_link_ratio_skew': 'skew'})
        feature = feature.merge(tmp_linkratio_agg, on='order_id', how='left')
        print('order link ratio sum mean max min var std finish!')
        #######################weather###################################################################
        weather = pd.read_csv(root_path+'weather.csv')
        weather_dict={'rainstorm':0,'heavy rain':1,'moderate rain':2,'cloudy':3,
                      'showers':4}
        weather['weather'] = weather['weather'].map(weather_dict)
        weather['date'] = weather['date'].astype(str)
        weather=weather[weather['date']==data_time]
        feature['weather'] = weather['weather'].values[0]
        feature['hightemp'] = weather['hightemp'].values[0]
        feature['lowtemp'] = weather['lowtemp'].values[0]
        print('weather finish!')
        ###################trend fea#############################################
        ###################trend link time#####################################
        data_link_split['link_current_status'] = data_link_split['link_current_status'].astype(int)
        groupby = data_link_split.groupby(['order_id'])
        func = partial(trend_in_last_k_link_id_time, periods=[2, 5, 7, 10, 15, 20, 30, 50, 80, 100, 100000000])
        g = parallel_apply_fea(groupby, func, index_name='order_id', num_workers=20, chunk_size=10000)
        feature = feature.merge(g, on='order_id', how='left')
        func = partial(last_link_time_features, periods=[2, 5, 7, 10, 15, 20, 30, 50, 80, 100, 100000000])
        g = parallel_apply_fea(groupby, func, index_name='order_id', num_workers=20, chunk_size=10000)
        feature = feature.merge(g, on='order_id', how='left')
        func = partial(last_k_link_time_interval, periods=[2, 5, 7, 10, 15, 20, 30, 50, 80, 100, 100000000])
        g = parallel_apply_fea(groupby, func, index_name='order_id', num_workers=20, chunk_size=10000)
        feature = feature.merge(g, on='order_id', how='left')
        print('trend link time finish!')
        ####################nextlinks graph embedding#######################
        data_link_split['link_id'] = data_link_split['link_id'].astype(int)
        data_link_split['link_id'] = data_link_split['link_id'].map(read_idkey)
        data_link_split['link_id'] = data_link_split['link_id'].fillna(0)
        data_link_split['link_id'] = data_link_split['link_id'].astype(int)
        data_link_split['link_id'] = data_link_split['link_id'].map(read_grapheb)
        data_link_split['link_id'] = data_link_split['link_id'].fillna('0')
        def replace_list(x):
            if isinstance(x, str):
                x = fill_list
            return x
        data_link_split['link_id'] = data_link_split['link_id'].apply(replace_list)
        link_id_col = ['zsl_link_id_eb{}'.format(i) for i in range(embedding_k)]
        agg_col = dict(zip(link_id_col, ['mean'] * len(link_id_col)))
        link_id_array = np.array(data_link_split.pop('link_id').to_list())
        link_id_array = pd.DataFrame(link_id_array, columns=agg_col, dtype=np.float16)
        data_link_split = pd.concat([data_link_split, link_id_array], axis=1)
        tmp = data_link_split.groupby('order_id', as_index=False)
        tmp_linkid_agg = tmp.agg(agg_col)
        feature = feature.merge(tmp_linkid_agg, on='order_id', how='left')

        feature.to_csv(root_path + 'feature/train/link_fea_order_id_level_{}.csv'.format(data_time), index=False)
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

    # link preprocess
    data_link = test[[1]]
    data_link['index'] = test_head.index
    data_link['order_id'] = test_head['order_id']
    data_link['ata'] = test_head['ata']
    data_link['distance'] = test_head['distance']
    data_link['simple_eta'] = test_head['simple_eta']
    data_link['slice_id'] = test_head['slice_id']

    # data_link['slice_id'] = data_link['slice_id'].apply(slice_id_change)
    gc.collect()
    data_link_split = data_link[1].str.split(' ', expand=True).stack().to_frame()
    data_link_split = data_link_split.reset_index(level=1, drop=True).rename(columns={0: 'link_info'})
    # data_link_split = data_link_split.reset_index(drop=True)
    data_link_split = data_link[['order_id', 'index', 'ata', 'distance', 'simple_eta', 'slice_id']].join(
        data_link_split)
    data_link_split = data_link_split.reset_index(drop=True)
    data_link_split[['link_id',
                     'link_time',
                     'link_ratio',
                     'link_current_status',
                     'link_arrival_status']] = data_link_split['link_info'].str.split(':|,', 5, expand=True)
    data_link_split = data_link_split.drop(['link_info'], axis=1)
    data_link_split['link_ratio'] = data_link_split['link_ratio'].astype(float)
    data_link_split['link_time'] = data_link_split['link_time'].astype(float)
    data_link_split['link_current_status'] = data_link_split['link_current_status'].astype(int)
    print('preprocess finish!')
    print('start feature engineering')
    feature = test_head[['order_id', 'distance']]
    ###################static fea#############################################
    #######################order link id count###############################
    df = data_link_split.groupby('order_id', as_index=False)
    tmp_linkid_agg = df['link_id'].agg({'zsl_order_link_id_count': 'count'})
    tmp_linkid_agg['zsl_order_link_id_count_bins'] = 0
    tmp_linkid_agg.loc[(tmp_linkid_agg['zsl_order_link_id_count'] >= 75) & (
                tmp_linkid_agg['zsl_order_link_id_count'] < 100), 'zsl_order_link_id_count_bins'] = 1
    tmp_linkid_agg.loc[(tmp_linkid_agg['zsl_order_link_id_count'] >= 100) & (
                tmp_linkid_agg['zsl_order_link_id_count'] < 120), 'zsl_order_link_id_count_bins'] = 2
    tmp_linkid_agg.loc[(tmp_linkid_agg['zsl_order_link_id_count'] >= 120), 'zsl_order_link_id_count_bins'] = 3
    feature = feature.merge(tmp_linkid_agg, on='order_id', how='left')
    print('order link id count finish!')
    #######################order link id & distance###############################
    feature['zsl_order_is_highspeed'] = 0
    feature.loc[
        (feature['distance'] > 90000) & (feature['zsl_order_link_id_count'] < 300), 'zsl_order_is_highspeed'] = 1
    print('order link id & distance finish!')
    #######################order link id & nextlinks centry###############################
    tmp = data_link_split[data_link_split['link_id'].isin(dc)]
    tmp = tmp.groupby('order_id', as_index=False)
    tmp_linkid_centry_count = tmp['link_id'].agg({'zsl_order_link_id_centry_count': 'count'})
    feature = feature.merge(tmp_linkid_centry_count, on='order_id', how='left')
    feature['zsl_order_link_id_centry_count'] = feature['zsl_order_link_id_centry_count'].fillna(0)
    print('order link id & nextlinks centry finish!')
    #######################order link time sum mean max min var std###############################
    tmp_linktime_agg = df['link_time'].agg({'zsl_order_link_time_sum': 'sum', 'zsl_order_link_time_mean': 'mean',
                                            'zsl_order_link_time_max': 'max', 'zsl_order_link_time_min': 'min',
                                            'zsl_order_link_time_var': 'var', 'zsl_order_link_time_skew': 'skew'})
    feature = feature.merge(tmp_linktime_agg, on='order_id', how='left')
    print('order link time sum mean max min var std finish!')
    #######################order link current status mean nunique###############################
    tmp_linktime_agg = df['link_current_status'].agg(
        {'zsl_link_current_status_mean': 'mean', 'zsl_link_current_status_nunique': 'nunique'})
    feature = feature.merge(tmp_linktime_agg, on='order_id', how='left')
    print('order link current status mean nunique finish!')
    #######################order link current status count vector###############################
    data_link_split['link_current_status'] = data_link_split['link_current_status'].astype(str)
    data_link_split.loc[data_link_split['link_current_status'].astype(int) < 0, 'link_current_status'] = '0'
    data_link_split.loc[data_link_split['link_current_status'].astype(int) > 3, 'link_current_status'] = '3'
    data = data_link_split.groupby('order_id')['link_current_status'].apply(lambda x: x.str.cat(sep=',')).reset_index()
    cv_encode = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    test_x = cv_encode.fit_transform(data['link_current_status'])
    test_x = test_x.toarray()
    link_current_status = pd.DataFrame(test_x, columns=['zsl_link_current_status0', 'zsl_link_current_status1',
                                                         'zsl_link_current_status2',
                                                         'zsl_link_current_status3'])
    data = pd.concat([data[['order_id']], link_current_status], axis=1)
    feature = feature.merge(data, on='order_id', how='left')
    print('order link current status count vector finish!')
    #######################order distance/link_id_count###############################
    feature['zsl_distance_div_link_id_count'] = feature['distance'] * 10 / feature['zsl_order_link_id_count']
    feature = feature.drop('distance', axis=1)
    print('order distance div link_id_count finish!')
    #######################order link ratio sum mean max min var std###############################
    tmp_linkratio_agg = df['link_ratio'].agg({'zsl_order_link_ratio_sum': 'sum', 'zsl_order_link_ratio_mean': 'mean',
                                              'zsl_order_link_ratio_min': 'min',
                                              'zsl_order_link_ratio_var': 'var', 'zsl_order_link_ratio_skew': 'skew'})
    feature = feature.merge(tmp_linkratio_agg, on='order_id', how='left')
    print('order link ratio sum mean max min var std finish!')
    #######################weather###################################################################
    weather = pd.read_csv(root_path + 'weather.csv')
    weather_dict = {'rainstorm': 0, 'heavy rain': 1, 'moderate rain': 2, 'cloudy': 3,
                    'showers': 4}
    weather['weather'] = weather['weather'].map(weather_dict)
    weather['date'] = weather['date'].astype(str)
    weather = weather[weather['date'] == data_time]
    feature['weather'] = weather['weather'].values[0]
    feature['hightemp'] = weather['hightemp'].values[0]
    feature['lowtemp'] = weather['lowtemp'].values[0]
    print('weather finish!')
    ###################trend fea#############################################
    ###################trend link time#####################################
    data_link_split['link_current_status'] = data_link_split['link_current_status'].astype(int)
    groupby = data_link_split.groupby(['order_id'])
    func = partial(trend_in_last_k_link_id_time, periods=[2, 5, 7, 10, 15, 20, 30, 50, 80, 100, 100000000])
    g = parallel_apply_fea(groupby, func, index_name='order_id', num_workers=20, chunk_size=10000)
    feature = feature.merge(g, on='order_id', how='left')
    func = partial(last_link_time_features, periods=[2, 5, 7, 10, 15, 20, 30, 50, 80, 100, 100000000])
    g = parallel_apply_fea(groupby, func, index_name='order_id', num_workers=20, chunk_size=10000)
    feature = feature.merge(g, on='order_id', how='left')
    func = partial(last_k_link_time_interval, periods=[2, 5, 7, 10, 15, 20, 30, 50, 80, 100, 100000000])
    g = parallel_apply_fea(groupby, func, index_name='order_id', num_workers=20, chunk_size=10000)
    feature = feature.merge(g, on='order_id', how='left')
    print('trend link time finish!')
    ####################nextlinks graph embedding#######################
    data_link_split['link_id'] = data_link_split['link_id'].astype(int)
    data_link_split['link_id'] = data_link_split['link_id'].map(read_idkey)
    data_link_split['link_id'] = data_link_split['link_id'].fillna(0)
    data_link_split['link_id'] = data_link_split['link_id'].astype(int)
    data_link_split['link_id'] = data_link_split['link_id'].map(read_grapheb)
    data_link_split['link_id'] = data_link_split['link_id'].fillna('0')
    def replace_list(x):
        if isinstance(x, str):
            x = fill_list
        return x
    data_link_split['link_id'] = data_link_split['link_id'].apply(replace_list)
    link_id_col = ['zsl_link_id_eb{}'.format(i) for i in range(embedding_k)]
    agg_col = dict(zip(link_id_col, ['mean'] * len(link_id_col)))
    link_id_array = np.array(data_link_split.pop('link_id').to_list())
    link_id_array = pd.DataFrame(link_id_array, columns=agg_col, dtype=np.float16)
    data_link_split = pd.concat([data_link_split, link_id_array], axis=1)
    tmp = data_link_split.groupby('order_id', as_index=False)
    tmp_linkid_agg = tmp.agg(agg_col)
    feature = feature.merge(tmp_linkid_agg, on='order_id', how='left')
    feature.to_csv(root_path+'feature/test/link_fea_order_id_level_20200901.csv',index=False)
