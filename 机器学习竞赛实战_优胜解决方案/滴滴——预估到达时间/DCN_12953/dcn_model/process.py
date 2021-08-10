import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
# import random
import gc
import ast
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
#pandarallel.initialize(nb_workers=16)
pandarallel.initialize()


def pandas_list_to_array(df):
    """
    Input: DataFrame of shape (x, y), containing list of length l
    Return: np.array of shape (x, l, y)
    """

    return np.transpose(
        np.array(df.values.tolist()),
        (0, 2, 1)
    )


def preprocess_inputs(df, cols: list):
    return pandas_list_to_array(
        df[cols]
    )


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


def load_data(making_data_dir, link_data_dir, cross_data_dir, link_data_other_dir, head_data_dir, 
              win_order_data_dir, pre_arrival_sqe_dir,zsl_link_data_dir, arrival_data_dir=None, zsl_arrival_data_dir=None, arrival_sqe_data_dir=None):
    """
    loading three path of data, then merge them
    :return: all data by order_level
    """
    print('-------------LOAD DATA for mk_data----------------')
    mk_list = file_name(making_data_dir)
    mk_list.sort()
    mk_data = append_all_data(mk_list, making_data_dir)
    #mk_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_order_xt/join_20200825.csv')  # for test running
    mk_data['date_time'] = mk_data['date_time'].astype(str)
    # print(mk_data['date_time'].head())
    mk_data['dayofweek'] = pd.to_datetime(mk_data['date_time'])
    mk_data['dayofweek'] = mk_data['dayofweek'].dt.dayofweek + 1
    weather_le = LabelEncoder()
    mk_data['weather_le'] = weather_le.fit_transform(mk_data['weather'])
    print('Remove the wk2_ and m1_')
    del_cols = []
    mk_cols = mk_data.columns.tolist()
    for i in range(len(mk_cols)):
        if 'wk2_' in mk_cols[i]:
            del_cols.append(mk_cols[i])
        if 'm1_' in mk_cols[i]:
            del_cols.append(mk_cols[i])
        if 'ratio' in mk_cols[i]:
            del_cols.append(mk_cols[i])
    del_cols = del_cols + ['weather', 'driver_id', 'date_time_dt', 'link_time_sum','date_time_sum']
    print('*-' * 40, 'Will be drop the list:', del_cols)
    mk_data.drop(columns=del_cols, axis=1, inplace=True)
    print('The init shape of mk_data:', mk_data.shape)
    #if arrival_data_dir:
    #    mk_data, _ = train_test_split(mk_data, test_size=0.4, random_state=42)
    #print('*-'*40)
    #print('The train_test_split shape of mk_data:', mk_data.shape)

    
    print('-------------LOAD WIN DATA----------------')
    win_order_list = file_name(win_order_data_dir)
    win_order_list.sort()
    win_order_data = append_all_data(win_order_list, win_order_data_dir)
    #win_order_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/win_order_xw/win_for_slice_20200825.csv')  # for test running
    del_win_order_cols = []
    win_order_cols = win_order_data.columns.tolist()
    for i in range(len(win_order_cols)):
        if 'last_wk_lk_current' in win_order_cols[i]:
            del_win_order_cols.append(win_order_cols[i])
        #if 'distance' in win_order_cols[i]:
        #    del_win_order_cols.append(win_order_cols[i])
        #if '1_percent' in win_order_cols[i]:
        #    del_win_order_cols.append(win_order_cols[i])
        #if '0_percent' in win_order_cols[i]:
        #    del_win_order_cols.append(win_order_cols[i])
    del_win_order_cols = del_win_order_cols + ['slice_id', 'date_time']
    win_order_data.drop(columns=del_win_order_cols, axis=1, inplace=True)
    print('win_order_data.shape',win_order_data.shape)
    mk_data = pd.merge(mk_data, win_order_data, how='left', on='order_id')
    print('mk_data.shape',mk_data.shape)
    del win_order_data
    gc.collect()


    """ 
    print('-------------LOAD ZSL DATA----------------')
    zsl_link_list = file_name(zsl_link_data_dir)
    zsl_link_list.sort()
    zsl_link_data = append_all_data(zsl_link_list, zsl_link_data_dir)
    #zsl_link_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/zsl_train_link/link_fea_order_id_level_20200825.csv')  # for test running
    get_zsl_link_cols = []
    zsl_link_cols = zsl_link_data.columns.tolist()
    for i in range(len(zsl_link_cols)):
        if 'eb' in zsl_link_cols[i]:
            get_zsl_link_cols.append(zsl_link_cols[i])
    #print(get_zsl_link_cols)
    get_zsl_link_cols.insert(0, 'order_id')
    print(zsl_link_data.shape)
    zsl_link_data = zsl_link_data[get_zsl_link_cols]
    print('mk_data.shape',mk_data.shape)
    mk_data = pd.merge(mk_data, zsl_link_data, on='order_id')
    print('mk_data.shape',mk_data.shape)
    del zsl_link_data
    gc.collect()
    """
    """
    #zsl_cross_list = file_name(zsl_cross_data_dir)
    #zsl_cross_list.sort()
    #zsl_cross_data = append_all_data(zsl_cross_list, zsl_cross_data_dir)
    zsl_cross_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/zsl_train_cross_0703/cross_fea_order_id_level_20200825.csv')  # for test running
    get_zsl_cross_cols = []
    zsl_cross_cols = zsl_cross_data.columns.tolist()
    for i in range(len(zsl_cross_cols)):
        if ('last' or 'div' or 'interval' or 'period') in zsl_cross_cols[i]:
            get_zsl_cross_cols.append(zsl_cross_cols[i])
    get_zsl_cross_cols.append('order_id')
    print(zsl_cross_data.shape)
    zsl_cross_data = zsl_cross_data[get_zsl_cross_cols]
    print('mk_data.shape',mk_data.shape)
    mk_data = pd.merge(mk_data, zsl_cross_data, on='order_id')
    print('mk_data.shape',mk_data.shape)
    del zsl_cross_data
    gc.collect()
    """
    
    print('-------------LOAD HEAD DATA----------------')
    head_list = file_name(head_data_dir)
    head_list.sort()
    head_data = append_all_data(head_list, head_data_dir)
    #head_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_head_link_data_clear/head_link_20200825.csv')  # for test running
    get_head_cols = ['len_tmp','status_0','status_1','status_2','status_3','status_4','rate_0','rate_1','rate_2','rate_3','rate_4']
    get_head_cols.insert(0, 'order_id')
    print('head_data.shape:',head_data.shape)
    head_data = head_data[get_head_cols]
    print('mk_data.shape',mk_data.shape)
    mk_data = pd.merge(mk_data, head_data, how='left', on='order_id')
    print('mk_data.shape',mk_data.shape)
    del head_data
    gc.collect()
    
    print('-------------LOAD DATA for link_data----------------')
    link_list = file_name(link_data_dir)
    link_list.sort()
    link_data = append_all_data(link_list, link_data_dir)
    # for test running
    #link_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_170_link_sqe_for_order/sqe_20200825_link.txt')
    print('The init shape of link_data:', link_data.shape)

    
    print('-------------LOAD DATA for arrival_sqe_data----------------')
    arrival_sqe_list = file_name(pre_arrival_sqe_dir)
    arrival_sqe_list.sort()
    arrival_sqe_data = append_all_data(arrival_sqe_list, pre_arrival_sqe_dir)
    #arrival_sqe_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/sqe_arrival_for_link/20200825.csv')  # for test running
    del arrival_sqe_data['slice_id']
    arrival_cols = arrival_sqe_data.columns.tolist()
    new_arrival_cols = ['future_'+i for i in arrival_cols if i != 'order_id']
    new_arrival_cols.insert(0, 'order_id')
    arrival_sqe_data.columns = new_arrival_cols
    print('The init shape of arrival_sqe_data:', arrival_sqe_data.shape)
    link_data = pd.merge(link_data, arrival_sqe_data, how='left', on='order_id')
    del arrival_sqe_data
    gc.collect()
    
    """
    print('-------------LOAD DATA for arrival_link_data----------------')
    arrival_link_list = file_name(pre_arrival_data_dir)
    arrival_link_list.sort()
    arrival_link_data = append_all_data(arrival_link_list, pre_arrival_data_dir)
    #arrival_link_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/final_pre_arrival_data/sqe_20200825_link.txt')  # for test running
    print('The init shape of arrival_link_data:', arrival_link_data.shape)
    link_data = pd.merge(link_data, arrival_link_data, how='left', on='order_id')
    del arrival_link_data
    gc.collect()
    """

    """
    print('-------------LOAD DATA for h_s_link_data----------------')
    h_s_link_list = file_name(h_s_for_link_dir)
    h_s_link_list.sort()
    h_s_link_data = append_all_data(h_s_link_list,h_s_for_link_dir)
    #h_s_link_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_hightmp_slice_for_link_eb/20200825_link.txt')  # for test running
    h_s_link_data = h_s_link_data[['order_id', 'sqe_slice_id', 'sqe_hightemp', 'sqe_weather_le']]
    print('The init shape of h_s_link_data:', h_s_link_data.shape)
    link_data = pd.merge(link_data, h_s_link_data, how='left', on='order_id')
    del h_s_link_data
    gc.collect()
    """
    print('-------------LOAD DATA for link_data_other----------------')
    link_list_other = file_name(link_data_other_dir)
    link_list_other.sort()
    link_data_other = append_all_data(link_list_other, link_data_other_dir)
    #link_data_other = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/for_0714_link_sqe_for_order_other/sqe_20200825_link.txt')  # for test running
    print('The init shape of link_data_other:', link_data_other.shape)

    link_data = pd.merge(link_data, link_data_other, on='order_id')
    # print(link_data.head(0))
    # del link_data['lk_t_sub_by_min']
    del_link_cols = ['lk_t_sub_by_min','lk_t_sub_by_q50', 'lk_t_sub_by_min', 'total_linktime_std']
                      # 'future_pre_arrival_status', 'future_arrive_slice_id']  # 'future_arrive_slice_id'
    link_data.drop(columns=del_link_cols, axis=1, inplace=True)
    print('The merge shape of link_data:', link_data.shape)
    del link_data_other
    gc.collect()

    print('-------------LOAD DATA for link_data_arrival----------------')
    if arrival_sqe_data_dir==None:
        pass
    else:
        link_list_arrival = file_name(arrival_sqe_data_dir)
        link_list_arrival.sort()
        link_data_arrival = append_all_data(link_list_arrival, arrival_sqe_data_dir)
        #link_data_arrival = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_170_lk_arrival_sqe_for_order/sqe_20200825_link.txt')  # for test running
        print('The init shape of link_data_arrival:', link_data_arrival.shape)
        link_data = pd.merge(link_data, link_data_arrival, on='order_id')
        print('The merge shape of link_data:', link_data.shape)
        del link_data_arrival
        gc.collect()

    link_cols_list = ['link_id', 'link_time', 'link_current_status', 'pr',
                      'dc', 'link_arrival_status', 'future_pre_arrival_status', 'future_arrive_slice_id']

    data = pd.merge(mk_data, link_data, how='left', on='order_id')
    del mk_data
    del link_data
    gc.collect()

    print('-------------LOAD DATA for arrival_data----------------')
    if arrival_data_dir==None:
        pass
    else:
        arrival_list = file_name(arrival_data_dir)
        arrival_list.sort()
        arrival_data = append_all_data(arrival_list, arrival_data_dir)
        #arrival_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_link_sqe_for_order_arrival/sqe_20200825_link.txt')
        arrival_cols = ['order_id', 'lk_arrival_0_percent', 'lk_arrival_1_percent','lk_arrival_2_percent', 'lk_arrival_3_percent', 'lk_arrival_4_percent']
        #print(arrival_data.head(2))
        data = pd.merge(data, arrival_data, how='left', on='order_id')
        del arrival_data
        gc.collect()
 
    print('-------------LOAD DATA for zsl_arrival_data----------------')
    if zsl_arrival_data_dir==None:
        pass
    else:
        zsl_arrival_list = file_name(zsl_arrival_data_dir)
        zsl_arrival_list.sort()
        zsl_arrival_data = append_all_data(zsl_arrival_list, zsl_arrival_data_dir)
        #zsl_arrival_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/zsl_arrival/link_fea_arrive_order_id_level_20200818.csv')
        zsl_arrival_cols = zsl_arrival_data.columns.tolist()
        zsl_arrival_cols.remove('order_id')
        #print(zsl_arrival_data.head(2))
        data = pd.merge(data, zsl_arrival_data, how='left', on='order_id')
        del zsl_arrival_data
        gc.collect()

    print('-------------LOAD DATA for cross_data----------------')
    cross_list = file_name(cross_data_dir)
    cross_list.sort()
    cross_data = append_all_data(cross_list, cross_data_dir)
    # for test running
    #cross_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/for_0714_cross_sqe_for_order/sqe_20200825_cross.txt')
    del_cross_cols = ['cr_t_sub_by_min', 'cr_t_sub_by_q50', 'total_crosstime_std']
    cross_data.drop(columns=del_cross_cols, axis=1, inplace=True)
    cross_cols_list = ['cross_id', 'cross_time']
    print('The init shape of cross_data:', cross_data.shape)

    data = pd.merge(data, cross_data, how='left', on='order_id')
    del cross_data
    gc.collect()
    # data['cross_id'] = data['cross_id'].str.replace('nan','0')
    # print('working..............................')

    mk_cols_list = data.columns.tolist()
    remove_mk_cols = ['order_id', 'slice_id', 'hightemp', 'lowtemp', 'weather_le', 'dayofweek', 'date_time', 'ata', 'link_arrival_status']
    mk_cols_list = list(set(mk_cols_list) - set(remove_mk_cols))
    mk_cols_list = list(set(mk_cols_list) - set(link_cols_list))
    mk_cols_list = list(set(mk_cols_list) - set(cross_cols_list))
    if arrival_data_dir==None:
        pass
    else:
        mk_cols_list = list(set(mk_cols_list) - set(arrival_cols))
        mk_cols_list = list(set(mk_cols_list) - set(zsl_arrival_cols))
    print('lenght of mk_cols_list', len(mk_cols_list))
    print('*-' * 40)
    print('The finish shape of data is:', data.shape)

    return data, mk_cols_list, link_cols_list, cross_cols_list


def processing_data(data, link_cols_list, cross_cols_list, mk_cols_list, WIDE_COLS, is_test=False):
    """
    fix data, ast.literal_eval + StandardScaler + train_test_split
    :return: train_data, val_data, test_data
    """
    #print('Now, Starting parallel_apply the arrival_status..................')
    #for i in tqdm(['link_arrival_status']):
    #    data[i] = data[i].parallel_apply(ast.literal_eval)
    print('Now, Starting parallel_apply the link..................')
    for i in tqdm(link_cols_list):
        data[i] = data[i].parallel_apply(ast.literal_eval)
    gc.collect()
    print('Now, Starting parallel_apply the cross..................')
    for i in tqdm(cross_cols_list):
        data[i] = data[i].parallel_apply(ast.literal_eval)
    data = data.fillna(0)

    # train, val
    if is_test is True:
        print('is_test is True')
        ss = joblib.load('../model_h5/ss_scaler')
        ss_cols = mk_cols_list + WIDE_COLS
        data[ss_cols] = ss.transform(data[ss_cols])
        return data
    else:
        ss_cols = mk_cols_list + WIDE_COLS
        ss = StandardScaler()
        ss.fit(data[ss_cols])
        data[ss_cols] = ss.transform(data[ss_cols])
        joblib.dump(ss, '../model_h5/ss_scaler')
        print('is_test is False')
        return data


def processing_inputs(data, mk_cols_list, link_cols_list, cross_cols_list, WIDE_COLS, arrival=True):
    """
    change the data for model
    :return:
    """
    print('*-'*40, processing_inputs)
    if arrival:
        mk_cols_list = mk_cols_list +  ['lk_arrival_0_percent', 'lk_arrival_1_percent','lk_arrival_2_percent', 'lk_arrival_3_percent', 'lk_arrival_4_percent']
        mk_cols_list = mk_cols_list + ['zsl_link_arrival_status_mean','zsl_link_arrival_status_nunique','zsl_link_arrival_status0','zsl_link_arrival_status1','zsl_link_arrival_status2','zsl_link_arrival_status3']
    if 'lk_arrival_0_percent' in mk_cols_list:
        print('The lk_arrival_0_percent in the mk_cols_list')
        #print('*-' * 40, 'EXIT')
        #sys.exit(0)
        print('111'*40, 'HAVE FEATURES OF ARRIVAL')
    else:
        print('222'*40, 'HAVENOT FEATURES OF ARRIVAL')
    if 'ata' in mk_cols_list:
        print('The ata in the mk_cols_list')
        print('*-' * 40, 'EXIT')
        sys.exit(0)
    if 'ata' in link_cols_list:
        print('The ata in the link_cols_list')
    if 'ata' in cross_cols_list:
        print('The ata in the cross_cols_list')
    if 'ata' in WIDE_COLS:
        print('The ata in the WIDE_COLS')
        print('*-' * 40, 'EXIT')
        sys.exit(0)
    data_link_inputs = preprocess_inputs(data, cols=link_cols_list)
    data.drop(columns=link_cols_list, axis=1, inplace=True)
    gc.collect()
    print('drop the link_cols_list')
    # print(data_link_inputs[:, :, :1])
    # data['cross_id'] = data['cross_id'].str.replace('nan','0')
    data_cross_inputs = preprocess_inputs(data, cols=cross_cols_list)
    data.drop(columns=cross_cols_list, axis=1, inplace=True)
    gc.collect()
    print('drop the cross_cols_list')

    data_deep_input = data[mk_cols_list]
    data_wide_input = data[WIDE_COLS].values
    data_inputs_slice = data['slice_id'].values
    data_labels = data['ata']
    if arrival:
        arrival_col = ['lk_arrival_0_percent', 'lk_arrival_1_percent',
                        'lk_arrival_2_percent', 'lk_arrival_3_percent', 'lk_arrival_4_percent']
        data_arrival = data[arrival_col]
        print('*-'*40, 'data_arrival', data_arrival.shape)
        return data_link_inputs, data_cross_inputs, data_deep_input, data_wide_input, data_inputs_slice, data_labels, data_arrival
    else:
        return data_link_inputs, data_cross_inputs, data_deep_input, data_wide_input, data_inputs_slice, data_labels


def split_col(data, columns, fillna=None):
    '''拆分成列

    :param data: 原始数据
    :param columns: 拆分的列名
    :type data: pandas.core.frame.DataFrame
    :type columns: list
    '''
    for c in columns:
        new_col = data.pop(c)
        max_len = max(list(map(lambda x:len(x) if isinstance(x, list) else 1, new_col.values)))  # 最大长度
        new_col = new_col.apply(lambda x: x+[fillna]*(max_len - len(x)) if isinstance(x, list) else [x]+[fillna]*(max_len - 1))  # 补空值，None可换成np.nan
        new_col = np.array(new_col.tolist()).T  # 转置
        for i, j in enumerate(new_col):
            data[c + str(i)] = j
    return data

def list_to_np(x):
    return np.array(x)


