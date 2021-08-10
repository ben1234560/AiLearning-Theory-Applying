import pandas as pd
import numpy as np
import gc
import tensorflow as tf
import process
import dcn_model
import sys
import random
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.random.set_seed(42)
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

RANDOM_SEED = 42
# types of columns of the data_set DataFrame
CATEGORICAL_COLS = [
    'weather_le', 'hightemp', 'lowtemp', 'dayofweek',
    'slice_id', 'link_current_status_4'
]

NUMERIC_COLS = [
    'distance', 'simple_eta', 'link_time_sum', 'link_count',
    'cr_t_sum', 'link_current_status_4_percent', 'link_current_status_mean',
    'pr_mean', 'dc_mean','lk_arrival_0_percent', 'lk_arrival_1_percent',
    'lk_arrival_2_percent', 'lk_arrival_3_percent', 'lk_arrival_4_percent'

]

WIDE_COLS = [
    'weather_le', 'hightemp', 'lowtemp', 'dayofweek'
]

IGNORE_COLS = [
    'order_id', 'ata'
]

TRAINING = True
VAL_TO_TEST = False


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    set_seed(RANDOM_SEED)
    print(dcn_model.get_available_gpus())  # 返回格式为：['/device:GPU:0', '/device:GPU:1']

    # LOAD DATA
    print('*-' * 40, 'LOAD DATA')
    making_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_order_xt/'
    link_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_170_link_sqe_for_order/'
    cross_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/for_0714_cross_sqe_for_order/'
    link_data_other_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/for_0714_link_sqe_for_order_other/'
    head_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_head_link_data_clear/'
    win_order_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/win_order_xw/'
    #pre_arrival_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/final_pre_arrival_data/'
    arrival_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_link_sqe_for_order_arrival/'
    zsl_arrival_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/zsl_arrival/'
    arrival_sqe_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_170_lk_arrival_sqe_for_order/'
    #h_s_for_link_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_hightmp_slice_for_link_eb/'
    pre_arrival_sqe_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/sqe_arrival_for_link/'
    zsl_link_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/zsl_train_link/'
    data, mk_cols_list, link_cols_list, cross_cols_list = process.load_data(making_data_dir,
                                                                            link_data_dir,
                                                                            cross_data_dir,
                                                                            link_data_other_dir,
                                                                            head_data_dir,
                                                                            win_order_data_dir,
                                                                            pre_arrival_sqe_dir,
                                                                            zsl_link_data_dir,
                                                                            #pre_arrival_data_dir,
                                                                            #h_s_for_link_dir,
                                                                            arrival_data_dir,
                                                                            zsl_arrival_data_dir,
                                                                            arrival_sqe_data_dir)
    
    #fd = dcn_model.FeatureDictionary(data, numeric_cols=NUMERIC_COLS, ignore_cols=IGNORE_COLS,
    #                                 cate_cols=CATEGORICAL_COLS)
    # PROCESSING DATA
    data['date_time'] = data['date_time'].astype(int)
    print("type(data['date_time']):", data['date_time'].dtype)
    data = data[data['date_time'] != 20200901]
    print('Here train_test_split..................')
    # all_train_data, _ = train_test_split(all_train_data, test_size=0.9, random_state=42)
    data = data.reset_index()
    del data['index']
    print('*-' * 40, 'The data.shape:', data.shape)
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=RANDOM_SEED)
    train_data = train_data.reset_index()
    val_data = val_data.reset_index()
    del train_data['index']
    del val_data['index']
    print('Save End.................')
    fb_list = CATEGORICAL_COLS+NUMERIC_COLS+IGNORE_COLS
    data_bak = data[fb_list]
    del data
    data = data_bak.copy()
    del data_bak
    gc.collect()

    print('*-' * 40, 'PROCESSING DATA FOR TRAIN')
    train_data = process.processing_data(train_data, link_cols_list, cross_cols_list, mk_cols_list, WIDE_COLS)
    #del data
    #fb_list = CATEGORICAL_COLS+NUMERIC_COLS+IGNORE_COLS
    #data = data[fb_list]
    #gc.collect()
    # print(train_data.columns.tolist())

    # PROCESSING INPUTS
    print('*-' * 40, 'PROCESSING INPUTS')
    # SAVE LIST
    a = np.array(mk_cols_list)
    np.save('../model_h5/mk_cols_list_0720_2.npy', a)
    a = np.array(link_cols_list)
    np.save('../model_h5/link_cols_list_0720_2.npy', a)
    a = np.array(cross_cols_list)
    np.save('../model_h5/cross_cols_list_0720_2.npy', cross_cols_list)
    a = np.array(CATEGORICAL_COLS)
    np.save('../model_h5/CATEGORICAL_COLS_0720_2.npy', a)
    del a
    pred_cols = ['ata']
    print('*-' * 40, 'PROCESSING INPUTS FOR TRAIN_DATA', train_data.shape)
    train_link_inputs, train_cross_inputs, train_deep_input, train_wide_input, \
        train_inputs_slice, train_labels, train_arrival = process.processing_inputs(
            train_data, mk_cols_list, link_cols_list, cross_cols_list, WIDE_COLS)
    X_train = dcn_model.preprocess(train_data, CATEGORICAL_COLS, NUMERIC_COLS)
    train_pre = train_data[['order_id']]
    del train_data
    gc.collect()

    print('*-' * 40, 'PROCESSING DATA FOR TRAIN')
    val_data = process.processing_data(val_data, link_cols_list, cross_cols_list, mk_cols_list, WIDE_COLS,  is_test=True)
    print('*-' * 40, 'PROCESSING INPUTS FOR VAL_DATA', val_data.shape)
    val_link_inputs, val_cross_inputs, val_deep_input, val_wide_input, \
        val_inputs_slice, val_labels, val_arrival = process.processing_inputs(
            val_data, mk_cols_list, link_cols_list, cross_cols_list, WIDE_COLS)
    X_val = dcn_model.preprocess(val_data, CATEGORICAL_COLS, NUMERIC_COLS)
    # val_data.to_csv('../model_h5/val_data.csv', index=0)  # saving csv for test running
    val_pre = val_data[['order_id']]
    del val_data
    gc.collect()

    # MODEL_INIT
    print('*-' * 40, 'T_MODEL_INIT')
    deep_col_len, wide_col_len = train_deep_input.values.shape[1], train_wide_input.shape[1]
    link_size = 639877 + 2
    cross_size = 44313 + 2
    link_nf_size, cross_nf_size = train_link_inputs.shape[2], train_cross_inputs.shape[2]
    slice_size = 288
    # link_seqlen, cross_seqlen = 170, 12  # 已默认
    print("link_size:{},link_nf_size:{},cross_size:{},cross_nf_size:{},slice_size:{}".format(link_size, link_nf_size,
                                                                                             cross_size, cross_nf_size,
                                                                                             slice_size))
    print("deep_col_len:{}, wide_col_len:{}".format(deep_col_len, wide_col_len))

    fd = dcn_model.FeatureDictionary(data, numeric_cols=NUMERIC_COLS, ignore_cols=IGNORE_COLS,
                                     cate_cols=CATEGORICAL_COLS)
    inp_layer, inp_embed = dcn_model.embedding_layers(fd)
    autoencoder, encoder = dcn_model.create_autoencoder(train_deep_input.values.shape[-1], 1, noise=0.1)
    if TRAINING:
        autoencoder.fit(train_deep_input.values, (train_deep_input.values, train_labels.values),
                        epochs=1000,  # 1000
                        batch_size=2048,  # 1024
                        validation_split=0.1,
                        callbacks=[tf.keras.callbacks.EarlyStopping('val_ata_output_loss', patience=10, restore_best_weights=True)])
        encoder.save_weights('../model_h5/t_encoder.hdf5')
    else:
        encoder.load_weights('../model_h5/t_encoder.hdf5')
    encoder.trainable = False
    del autoencoder

    t_model = dcn_model.DCN_model(inp_layer, inp_embed, link_size, cross_size, slice_size, deep_col_len, wide_col_len,
                                link_nf_size, cross_nf_size, encoder, conv=True, have_knowledge=False)    
    #del encoder
    gc.collect()
    
    mc, es, lr = dcn_model.get_mc_es_lr('0720_2', patience=5, min_delta=1e-4)
    print('*-' * 40, 'MODEL_INIT END')
   
    print('*-' * 40, 'ARRIVAL_MODEL_FIT')
    t_history = t_model.fit(
        [
            X_train['weather_le'], X_train['hightemp'], X_train['lowtemp'], X_train['dayofweek'],
            X_train['slice_id'], X_train['link_current_status_4'],
            X_train['distance'], X_train['simple_eta'], X_train['link_time_sum'], X_train['link_count'],
            X_train['cr_t_sum'], X_train['link_current_status_4_percent'], X_train['link_current_status_mean'],
            X_train['pr_mean'], X_train['dc_mean'],
            X_train['lk_arrival_0_percent'], X_train['lk_arrival_1_percent'],X_train['lk_arrival_2_percent'], 
            X_train['lk_arrival_3_percent'],X_train['lk_arrival_4_percent'],
            train_link_inputs, train_cross_inputs, train_deep_input.values, train_wide_input, train_inputs_slice],
        train_labels.values,
        validation_data=(
            [
                X_val['weather_le'], X_val['hightemp'], X_val['lowtemp'], X_val['dayofweek'],
                X_val['slice_id'], X_val['link_current_status_4'],
                X_val['distance'], X_val['simple_eta'], X_val['link_time_sum'], X_val['link_count'],
                X_val['cr_t_sum'], X_val['link_current_status_4_percent'], X_val['link_current_status_mean'],
                X_val['pr_mean'], X_val['dc_mean'],
                X_val['lk_arrival_0_percent'], X_val['lk_arrival_1_percent'],X_val['lk_arrival_2_percent'], 
                X_val['lk_arrival_3_percent'],X_val['lk_arrival_4_percent'],
                val_link_inputs, val_cross_inputs, val_deep_input.values, val_wide_input, val_inputs_slice],
                (val_labels.values),),
        batch_size=2048,  # 2048,1024
        epochs=100,  # 100
        verbose=1,
        # )
        callbacks=[es])  # lr
    np.save('../model_h5/t_model_0720_2.npy', t_history.history)
    t_model.save_weights("../model_h5/t_model_0720_2.h5")
    print('*-' * 40, 't_MODEL_PREDICT')
    y_knowledge_train = t_model.predict(
            [X_train['weather_le'], X_train['hightemp'], X_train['lowtemp'], X_train['dayofweek'],
            X_train['slice_id'], X_train['link_current_status_4'],
            X_train['distance'], X_train['simple_eta'], X_train['link_time_sum'], X_train['link_count'],
            X_train['cr_t_sum'], X_train['link_current_status_4_percent'], X_train['link_current_status_mean'],
            X_train['pr_mean'], X_train['dc_mean'],
            X_train['lk_arrival_0_percent'], X_train['lk_arrival_1_percent'],X_train['lk_arrival_2_percent'], 
            X_train['lk_arrival_3_percent'],X_train['lk_arrival_4_percent'],
            train_link_inputs, train_cross_inputs, train_deep_input.values, train_wide_input, train_inputs_slice],
            batch_size=2048)
    y_knowledge_val = t_model.predict(
            [
                X_val['weather_le'], X_val['hightemp'], X_val['lowtemp'], X_val['dayofweek'],
                X_val['slice_id'], X_val['link_current_status_4'],
                X_val['distance'], X_val['simple_eta'], X_val['link_time_sum'], X_val['link_count'],
                X_val['cr_t_sum'], X_val['link_current_status_4_percent'], X_val['link_current_status_mean'],
                X_val['pr_mean'], X_val['dc_mean'],
                X_val['lk_arrival_0_percent'], X_val['lk_arrival_1_percent'],X_val['lk_arrival_2_percent'],
                X_val['lk_arrival_3_percent'],X_val['lk_arrival_4_percent'],
                val_link_inputs, val_cross_inputs, val_deep_input.values, val_wide_input, val_inputs_slice],
                batch_size=2048)
    print('*-'*40, 'TRAINFORME')
    train_labels = pd.DataFrame(train_labels)
    train_labels['y_knowledge_train'] = np.squeeze(y_knowledge_train)
    print(np.squeeze(y_knowledge_train)[:2])
    print(train_labels['y_knowledge_train'].head(2))
    val_labels = pd.DataFrame(val_labels) 
    val_labels['y_knowledge_val'] = np.squeeze(y_knowledge_val)
    print('*-' * 40, 't_MODEL_END')
    zsl_arrival_cols = ['zsl_link_arrival_status_mean','zsl_link_arrival_status_nunique','zsl_link_arrival_status0','zsl_link_arrival_status1','zsl_link_arrival_status2','zsl_link_arrival_status3']
    train_deep_input = train_deep_input.drop(['lk_arrival_0_percent','lk_arrival_1_percent','lk_arrival_2_percent','lk_arrival_3_percent','lk_arrival_4_percent'],axis=1)
    train_deep_input = train_deep_input.drop(zsl_arrival_cols, axis=1)

    val_deep_input = val_deep_input.drop(['lk_arrival_0_percent','lk_arrival_1_percent','lk_arrival_2_percent','lk_arrival_3_percent','lk_arrival_4_percent'],axis=1)
    val_deep_input = val_deep_input.drop(zsl_arrival_cols, axis=1)

    if 'ata' in train_deep_input.columns.tolist():
        print('The ata in the train_deep_input')
        print('*-' * 40, 'EXIT')
        sys.exit(0)
    if 'lk_arrival_0_percent' in train_deep_input.columns.tolist():
        print('The lk_arrival_0_percent in the train_deep_input')
        print('*-' * 40, 'EXIT')
        sys.exit(0)
    if 'lk_arrival_0_percent' in val_deep_input.columns.tolist():
        print('The lk_arrival_0_percent in the val_deep_input')
        print('*-' * 40, 'EXIT')
        sys.exit(0)
    if 'zsl_link_arrival_status_mean' in train_deep_input.columns.tolist():
        print('The zsl_link_arrival_status_mean in the train_deep_input')
        print('*-' * 40, 'EXIT')
        sys.exit(0)

    mk_cols_list = train_deep_input.columns.tolist()
    print('*-' * 40, 'MODEL_FIT')
    deep_col_len, wide_col_len = train_deep_input.values.shape[1], train_wide_input.shape[1]
    print("deep_col_len:{}, wide_col_len:{}".format(deep_col_len, wide_col_len))
    NUMERIC_COLS = list(set(NUMERIC_COLS)-set(['lk_arrival_0_percent','lk_arrival_1_percent','lk_arrival_2_percent',
                                     'lk_arrival_3_percent','lk_arrival_4_percent']))
    fb_list = CATEGORICAL_COLS+NUMERIC_COLS+IGNORE_COLS
    if 'lk_arrival_0_percent' in fb_list:
        print('The lk_arrival_0_percent in the fb_list')
        print('*-' * 40, 'EXIT')
        sys.exit(0)
    data = data[fb_list]
    fd = dcn_model.FeatureDictionary(data, numeric_cols=NUMERIC_COLS, ignore_cols=IGNORE_COLS,
                                     cate_cols=CATEGORICAL_COLS)
    inp_layer, inp_embed = dcn_model.embedding_layers(fd)
    autoencoder, encoder = dcn_model.create_autoencoder(train_deep_input.values.shape[-1], 1, noise=0.1)
    if TRAINING:
        autoencoder.fit(train_deep_input.values, (train_deep_input.values, train_labels['ata'].values),
                        epochs=1000,  # 1000
                        batch_size=2048,  # 1024
                        validation_split=0.1,
                        callbacks=[tf.keras.callbacks.EarlyStopping('val_ata_output_loss', patience=10, restore_best_weights=True)])
        encoder.save_weights('../model_h5/main_encoder.hdf5')
    else:
        encoder.load_weights('../model_h5/main_encoder.hdf5')
    encoder.trainable = False
    del autoencoder
  
    #print(type(train_labels['y_knowledge_train']))
    #print(type(train_labels))
    #y_train = np.vstack((train_labels, train_pre['y_knowledge_train'])).T
    #y_valid = np.vstack((val_labels, val_pre['y_knowledge_val'])).T
    #print(train_labels.shape)
    print(train_labels.head(1))
    print(train_labels.values[0])

    print('*-'*40, 'The shape of train_link_inputs before', train_link_inputs.shape)
    train_link_inputs = np.concatenate((train_link_inputs[:, :, :5], train_link_inputs[:, :, 6:]), axis=2)
    
    print('*-'*40, 'The shape of train_link_inputs after', train_link_inputs.shape)
    val_link_inputs = np.concatenate((val_link_inputs[:, :, :5], val_link_inputs[:, :, 6:]), axis=2)
    link_nf_size, cross_nf_size = train_link_inputs.shape[2], train_cross_inputs.shape[2]
    mc, es, lr = dcn_model.get_mc_es_lr_for_student('0720_2', patience=5, min_delta=1e-4)
    model = dcn_model.DCN_model(inp_layer, inp_embed, link_size, cross_size, slice_size, deep_col_len, wide_col_len,
                                link_nf_size, cross_nf_size, encoder, conv=True)
    history = model.fit(
        [
            X_train['weather_le'], X_train['hightemp'], X_train['lowtemp'], X_train['dayofweek'],
            X_train['slice_id'], X_train['link_current_status_4'],
            X_train['distance'], X_train['simple_eta'], X_train['link_time_sum'], X_train['link_count'],
            X_train['cr_t_sum'], X_train['link_current_status_4_percent'], X_train['link_current_status_mean'],
            X_train['pr_mean'], X_train['dc_mean'],
            train_link_inputs, train_cross_inputs, train_deep_input.values, train_wide_input, train_inputs_slice],
        train_labels.values,
        validation_data=(
            [
                X_val['weather_le'], X_val['hightemp'], X_val['lowtemp'], X_val['dayofweek'],
                X_val['slice_id'], X_val['link_current_status_4'],
                X_val['distance'], X_val['simple_eta'], X_val['link_time_sum'], X_val['link_count'],
                X_val['cr_t_sum'], X_val['link_current_status_4_percent'], X_val['link_current_status_mean'],
                X_val['pr_mean'], X_val['dc_mean'],
                val_link_inputs, val_cross_inputs, val_deep_input.values, val_wide_input, val_inputs_slice], 
                (val_labels.values),),
        batch_size=2048,  # 2048,1024
        epochs=100,  # 100
        verbose=1,
        # )
        callbacks=[es])  # lr
    np.save('../model_h5/history_0720_2.npy', history.history)
    model.save_weights("../model_h5/dcn_model_0720_2.h5")
    # MODEL_RPEDICT
    if VAL_TO_TEST:
        print('*-'*40,'val_to_test')
        val_pre = val_pre.rename(columns={'order_id': 'id'})
        print(val_link_inputs.shape, val_cross_inputs.shape, X_val.shape)
        print('*-' * 40, 'MODEL_RPEDICT')
        val_pred = model.predict(
            [
                X_val['weather_le'], X_val['hightemp'], X_val['lowtemp'], X_val['dayofweek'],
                X_val['slice_id'], X_val['link_current_status_4'],
                X_val['distance'], X_val['simple_eta'], X_val['link_time_sum'], X_val['link_count'],
                X_val['cr_t_sum'], X_val['link_current_status_4_percent'], X_val['link_current_status_mean'],
                X_val['pr_mean'], X_val['dc_mean'],
                val_link_inputs, val_cross_inputs, val_deep_input.values, val_wide_input, val_inputs_slice],
               batch_size=2048)
        val_pre['val_predict'] = np.squeeze(val_pred[:, 1])
        val_pre['other_predict'] = np.squeeze(val_pred[:, 0])
        # val_pre['val_predict'] = val_pre['val_predict'].round(0)
        val_pre = val_pre.rename(columns={'val_predict': 'result'})  # 更改列名
        val_pre = val_pre[['id', 'result', 'other_predict']]
        val_pre['ata'] = val_labels['ata'].values
        print(val_pre.head())
        result_save_path = '../result_csv/val_0720_2.csv'
        print('*-' * 40, 'CSV_SAVE_PATH:', result_save_path)
        print('..........Finish')

    del X_train, train_link_inputs, train_cross_inputs, train_deep_input, \
        train_wide_input, train_inputs_slice, train_labels
    del X_val, val_link_inputs, val_cross_inputs, val_deep_input, val_wide_input, val_inputs_slice, val_labels
    gc.collect()
    #print('*-' * 40, 'EXIT')
    #sys.exit(0)
    print('*-' * 40, 'LOAD TEST DATA')
    making_test_data_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/order_xt/'
    link_test_data_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/max_170_link_sqe_for_order/'
    cross_test_data_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/cross_sqe_for_order/'
    link_test_data_other_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/link_sqe_for_order_other/'
    head_test_data_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/head_link_data_clear/'
    win_order_test_data_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/win_order_xw/'
    pre_arrival_sqe_test_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/sqe_arrival_for_link/'
    #h_s_for_test_link_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/max_hightmp_slice_for_link_eb/'
    #pre_arrival_test_data_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/final_pre_arrival_data/'
    zsl_link_test_data_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/zsl_test_link/'
    #zsl_cross_test_data_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/zsl_test_cross_0703/'
    test_data, _, _, _ = process.load_data(making_test_data_dir,
                                                                                 link_test_data_dir,
                                                                                 cross_test_data_dir,
                                                                                 link_test_data_other_dir,
                                                                                 head_test_data_dir,
                                                                                 win_order_test_data_dir,
                                                                                 pre_arrival_sqe_test_dir,
                                                                                 zsl_link_test_data_dir) #,
                                                                                 #h_s_for_test_link_dir)
                                                                                 #pre_arrival_test_data_dir)
    print('*-' * 40, 'PROCESSING DATA')
    link_cols_list.remove('link_arrival_status')
    test_data = process.processing_data(test_data, link_cols_list, cross_cols_list, mk_cols_list, WIDE_COLS, is_test=True)
    gc.collect()
    print('*-' * 40, 'PROCESSING INPUTS FOR TEST_DATA', test_data.shape)
    test_link_inputs, test_cross_inputs, test_deep_input, test_wide_input, \
        test_inputs_slice, _ = process.processing_inputs(
            test_data, mk_cols_list, link_cols_list, cross_cols_list, WIDE_COLS, arrival=False)
    X_test = dcn_model.preprocess(test_data, CATEGORICAL_COLS, NUMERIC_COLS)
    test_pre = test_data[['order_id']]
    test_arrival_pre = test_data[['order_id']]
    gc.collect()

    test_pre = test_pre.rename(columns={'order_id': 'id'})
    print(test_link_inputs.shape, test_cross_inputs.shape, X_test.shape, test_deep_input.shape)
    print('*-' * 40, 'MODEL_RPEDICT')
    test_pred = model.predict(
        [
            X_test['weather_le'], X_test['hightemp'], X_test['lowtemp'], X_test['dayofweek'],
            X_test['slice_id'], X_test['link_current_status_4'],
            X_test['distance'], X_test['simple_eta'], X_test['link_time_sum'], X_test['link_count'],
            X_test['cr_t_sum'], X_test['link_current_status_4_percent'], X_test['link_current_status_mean'],
            X_test['pr_mean'], X_test['dc_mean'],
            test_link_inputs, test_cross_inputs, test_deep_input.values, test_wide_input, test_inputs_slice],
           batch_size=2048)
    test_pre['test_predict'] = np.squeeze(test_pred[:, 1])
    test_pre['other_predict'] = np.squeeze(test_pred[:, 0])
    # test_pre['test_predict'] = test_pre['test_predict'].round(0)
    test_pre = test_pre.rename(columns={'test_predict': 'result'})  # 更改列名
    test_pre = test_pre[['id', 'result','other_predict']]
    print(test_pre.head())
    result_save_path = '../result_csv/submit_0720_2.csv'
    print('*-' * 40, 'CSV_SAVE_PATH:', result_save_path)
    test_pre.to_csv(result_save_path, index=0)  # 保存

    print('..........Finish')
