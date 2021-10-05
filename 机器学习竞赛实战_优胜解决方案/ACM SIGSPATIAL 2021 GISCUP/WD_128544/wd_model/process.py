import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
# import random
import gc
import ast
import os
import warnings
import joblib


warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
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


def load_data(making_data_dir, link_data_dir, cross_data_dir, head_link_dir,
             win_order_data_dir, pre_arrival_sqe_dir, data_for_driver_xw, downstream_status_dir):
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
    mk_data['dayofweek'] = pd.to_datetime(mk_data['date_time'])
    mk_data['dayofweek'] = mk_data['dayofweek'].dt.dayofweek+1

    weather_le = LabelEncoder()
    mk_data['weather_le'] = weather_le.fit_transform(mk_data['weather'])
    mk_data['driver_id'] = mk_data['driver_id'].astype(str)

    """
    print('-------------LOAD DATA for driver_data----------------')
    driver_list = file_name(data_for_driver_xw)
    driver_list.sort()
    driver_data = append_all_data(driver_list, data_for_driver_xw)
    #driver_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/data_for_driver_xw/driver_20200825_head.txt')
    driver_data = driver_data[['driver_id','date_time','entropy','hour_mean','workday_order','weekend_order']]
    driver_data['date_time'] = driver_data['date_time'].astype(str)
    driver_data['driver_id'] = driver_data['driver_id'].astype(str)
    mk_data = mk_data.merge(driver_data, on=['driver_id', 'date_time'], how='left')
    del driver_data
    """

    """
    print('-------------LOAD DATA for downstream_status_for_order----------------')
    ds_data_list = file_name(downstream_status_dir)
    ds_data_list.sort()
    ds_link_data = append_all_data(ds_data_list, downstream_status_dir)
    #ds_link_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/downstream_status_for_order/ds_for_order_20200825.csv')
    mk_data = mk_data.merge(ds_link_data, on=['order_id'], how='left')
    del ds_link_data
    """


    """
    print('-------------LOAD DATA for rate_status_for_order----------------')
    #rate_data_list = file_name(rate_status_for_order)
    #rate_data_list.sort()
    #rate_data = append_all_data(rate_data_list, rate_status_for_order)
    rate_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/rate_status_for_order/rate_for_order_20200825.csv')
    mk_data = mk_data.merge(rate_data, on=['order_id'], how='left')
    del rate_data
    """


    print('Remove the wk2_ and m1_ and ratio')
    del_cols = []
    mk_cols = mk_data.columns.tolist()
    for i in range(len(mk_cols)):
        if 'wk2_' in mk_cols[i]:
            del_cols.append(mk_cols[i])
        if 'm1_' in mk_cols[i]:
            del_cols.append(mk_cols[i])
        if 'ratio' in mk_cols[i]:
            del_cols.append(mk_cols[i])
    del_cols = del_cols + ['date_time_mean','weather', 'driver_id', 'date_time_dt', 'link_time_sum','date_time_sum']
    print('*-' * 40, 'Will be drop the list:', del_cols)
    mk_data.drop(columns=del_cols, axis=1, inplace=True)
    print('The init shape of mk_data:', mk_data.shape)


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


    print('-------------LOAD HEAD DATA----------------')
    head_list = file_name(head_link_dir)
    head_list.sort()
    head_data = append_all_data(head_list, head_link_dir)
    #head_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/head_link_data_clear/head_link_20200825.csv')  # for test running
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
    #link_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_170_link_sqe_for_order/sqe_20200825_link.txt')  # for test running
    #del_link_cols = ['link_time_sub','link_time_sub_sum','link_time_sub_mean', 'link_time_sub_std','link_time_sub_skew']
    #link_data.drop(del_link_cols, axis=1, inplace=True)
    print('The init shape of link_data:', link_data.shape)
    gc.collect()


    print('-------------LOAD DATA for arrival_sqe_data----------------')
    arrival_sqe_list = file_name(pre_arrival_sqe_dir)
    arrival_sqe_list.sort()
    arrival_sqe_data = append_all_data(arrival_sqe_list, pre_arrival_sqe_dir)
    #arrival_sqe_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/sqe_arrival_for_link/20200825.csv')  # for test running
    del arrival_sqe_data['slice_id']
    del arrival_sqe_data['pre_arrival_status']
    del arrival_sqe_data['arrive_slice_id']
    arrival_cols = arrival_sqe_data.columns.tolist()
    new_arrival_cols = ['future_'+i for i in arrival_cols if i != 'order_id']
    new_arrival_cols.insert(0, 'order_id')
    arrival_sqe_data.columns = new_arrival_cols
    print('The init shape of arrival_sqe_data:', arrival_sqe_data.shape)
    link_data = pd.merge(link_data, arrival_sqe_data, how='left', on='order_id')
    del arrival_sqe_data
    gc.collect()
    link_cols_list = ['link_id', 'link_time', 'link_current_status', 'pr','dc']



    print('-------------LOAD DATA for cross_data----------------')
    cross_list = file_name(cross_data_dir)
    cross_list.sort()
    cross_data = append_all_data(cross_list, cross_data_dir)
    #cross_data = pd.read_csv('/home/didi2021/didi2021/giscup_2021/final_train_data_0703/for_0714_cross_sqe_for_order/sqe_20200825_cross.txt')  # for test running
    del_cross_cols = ['cr_t_sub_by_min', 'cr_t_sub_by_q50', 'total_crosstime_std']
    cross_data.drop(columns=del_cross_cols, axis=1, inplace=True)
    print('The init shape of cross_data:', cross_data.shape)
    cross_cols_list = ['cross_id', 'cross_time']


    data = pd.merge(mk_data, link_data, how='left', on='order_id')
    del mk_data
    del link_data
    gc.collect()
    data = pd.merge(data, cross_data, how='left', on='order_id')
    del cross_data
    gc.collect()

    # remove the class type and id and label, for deep inputs
    mk_cols_list = data.columns.tolist()
    remove_mk_cols = ['order_id', 'slice_id', 'hightemp', 'lowtemp', 'weather_le', 'dayofweek', 'date_time', 'ata']
    mk_cols_list = list(set(mk_cols_list) - set(remove_mk_cols))
    mk_cols_list = list(set(mk_cols_list) - set(link_cols_list))
    mk_cols_list = list(set(mk_cols_list) - set(cross_cols_list))
    print('lenght of mk_cols_list', len(mk_cols_list))
    print('*-' * 40)
    print('The finish shape of data is:', data.shape)

    return data, mk_cols_list, link_cols_list, cross_cols_list


def processing_data(data, mk_cols_list, link_cols_list, cross_cols_list, WIDE_COLS, is_test=False):
    """
    fix data, ast.literal_eval + StandardScaler + train_test_split
    :return: train_data, val_data, test_data
    """
    print('Now, Starting parallel_apply the link..................')
    for i in tqdm(link_cols_list):
        data[i] = data[i].parallel_apply(ast.literal_eval)
    print('Now, Starting parallel_apply the cross..................')
    for i in tqdm(cross_cols_list):
        data[i] = data[i].parallel_apply(ast.literal_eval)
    # data = data.fillna(0)
    data.fillna(data.median(),inplace=True)
    ss_cols = mk_cols_list + WIDE_COLS
    
        # train, val
    if is_test is True:
        print('is_test is True')
        ss = joblib.load('../model_h5/ss_scaler')
        data[ss_cols] = ss.transform(data[ss_cols])
        return data
    else:
        ss = StandardScaler()
        ss.fit(data[ss_cols])
        data[ss_cols] = ss.transform(data[ss_cols])
        joblib.dump(ss, '../model_h5/ss_scaler')
        print('is_test is False')
        data['date_time'] = data['date_time'].astype(int)
        print("type(data['date_time']):", data['date_time'].dtype)
        # print('Here train_test_split..................')
        # all_train_data, _ = train_test_split(all_train_data, test_size=0.9, random_state=42)
        print('*-' * 40, 'The data.shape:', data.shape)
        train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)
        train_data = train_data.reset_index()
        val_data = val_data.reset_index()
        del train_data['index']
        del val_data['index']
        return train_data, val_data


def processing_inputs(data, mk_cols_list, link_cols_list, cross_cols_list, WIDE_COLS):
    """
    change the data for model
    :return:
    """
    if 'ata' in mk_cols_list:
        print('The ata in the mk_cols_list')
    if 'ata' in link_cols_list:
        print('The ata in the link_cols_list')
    if 'ata' in cross_cols_list:
        print('The ata in the cross_cols_list')
    if 'ata' in WIDE_COLS:
        print('The ata in the WIDE_COLS')
    #link_cols_list = ['link_id', 'link_time','link_id_count','pr','dc',
    #                                                   'top_a','link_current_status','link_ratio']
    #cross_cols_list = ['cross_id', 'cross_time']
    data_link_inputs = preprocess_inputs(data, cols=link_cols_list)
    data_cross_inputs = preprocess_inputs(data, cols=cross_cols_list)
    data_deep_input = data[mk_cols_list].values
    data_wide_input = data[WIDE_COLS].values
    data_inputs_slice = data['slice_id'].values
    # print('--------------------------------test, ', min(data['slice_id'].values.tolist()))
    data_labels = data['ata'].values

    return data_link_inputs, data_cross_inputs, data_deep_input, data_wide_input, data_inputs_slice, data_labels
