import pandas as pd
import numpy as np
import gc
import process
import wd_model
import time


RANDOM_SEED = 42

# types of columns of the data_set DataFrame
WIDE_COLS = [
    'weather_le', 'hightemp', 'lowtemp', 'dayofweek'
]

if __name__ == '__main__':
    t1 = time.time()
    print(wd_model.get_available_gpus())  # 返回格式为：['/device:GPU:0', '/device:GPU:1']

    # LOAD DATA
    print('*-' * 40, 'LOAD DATA')
    making_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_order_xt/'
    link_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_170_link_sqe_for_order/'
    cross_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/for_0714_cross_sqe_for_order/'
    head_link_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/max_head_link_data_clear/'
    win_order_data_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/win_order_xw/'
    pre_arrival_sqe_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/sqe_arrival_for_link/'
    data_for_driver_xw = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/data_for_driver_xw/'
    downstream_status_dir = '/home/didi2021/didi2021/giscup_2021/final_train_data_0703/downstream_status_for_order/'
    data, mk_cols_list, link_cols_list, cross_cols_list = process.load_data(making_data_dir,
                                                                            link_data_dir,
                                                                            cross_data_dir,
                                                                            head_link_dir,
                                                                            win_order_data_dir,
                                                                            pre_arrival_sqe_dir,
                                                                            data_for_driver_xw,
                                                                            downstream_status_dir)

    # PROCESSING DATA
    print('*-' * 40, 'PROCESSING DATA')
    train_data, val_data = process.processing_data(data, mk_cols_list, link_cols_list, cross_cols_list,
                                                              WIDE_COLS)
    del data
    gc.collect()
    # print(train_data.columns.tolist())

    # PROCESSING INPUTS
    print('*-' * 40, 'PROCESSING INPUTS')
    # SAVE LIST
    a = np.array(mk_cols_list)
    np.save('../model_h5/wd_mk_cols_list_0730_5.npy', a)
    a = np.array(link_cols_list)
    np.save('../model_h5/wd_link_cols_list_0730_5.npy', a)
    a = np.array(cross_cols_list)
    np.save('../model_h5/wd_cross_cols_list_0730_5.npy', cross_cols_list)
    pred_cols = ['ata']
    print('*-' * 40, 'PROCESSING INPUTS FOR TRAIN_DATA', train_data.shape)
    train_link_inputs, train_cross_inputs, train_deep_input, train_wide_input, \
    train_inputs_slice, train_labels = process.processing_inputs(
        train_data, mk_cols_list, link_cols_list, cross_cols_list, WIDE_COLS)
    del train_data
    gc.collect()

    print('*-' * 40, 'PROCESSING INPUTS FOR VAL_DATA', val_data.shape)
    val_link_inputs, val_cross_inputs, val_deep_input, val_wide_input, \
    val_inputs_slice, val_labels = process.processing_inputs(
        val_data, mk_cols_list, link_cols_list, cross_cols_list, WIDE_COLS)
    del val_data
    gc.collect()


    # MODEL_INIT
    print('*-' * 40, 'MODEL_INIT')
    deep_col_len, wide_col_len = train_deep_input.shape[1], train_wide_input.shape[1]
    link_nf_size, cross_nf_size = train_link_inputs.shape[2], train_cross_inputs.shape[2]
    link_size = 639877 + 2
    cross_size = 44313 + 2
    slice_size = 288
    # link_seqlen, cross_seqlen = 170, 12  # 已默认
    print("link_size:{},link_nf_size:{},cross_size:{},cross_nf_size:{},slice_size:{}".format(link_size, link_nf_size,
                                                                                             cross_size, cross_nf_size,
                                                                                             slice_size))
    print("deep_col_len:{}, wide_col_len:{}".format(deep_col_len, wide_col_len))

    model = wd_model.wd_model(link_size, cross_size, slice_size, deep_col_len, wide_col_len,
                              link_nf_size, cross_nf_size, conv='conv')

    mc, es, lr = wd_model.get_mc_es_lr('0730_5', patience=4, min_delta=1e-4)
    print('*-' * 40, 'MODEL_INIT END')
    # MODEL_FIT
    print('*-' * 40, 'MODEL_FIT_PREDICT')
    history = model.fit(
        [train_link_inputs, train_cross_inputs, train_deep_input, train_wide_input, train_inputs_slice], train_labels,
        validation_data=(
            [val_link_inputs, val_cross_inputs, val_deep_input, val_wide_input, val_inputs_slice], val_labels),
        batch_size=2048,  # 2048,256
        epochs=100,
        verbose=1,
        callbacks=[es])
    np.save('../model_h5/history_0730_5.npy', history.history)
    model.save_weights("../model_h5/wd_model_0730_5.h5")

    del train_link_inputs, train_cross_inputs, train_deep_input, \
        train_wide_input, train_inputs_slice, train_labels
    del val_link_inputs, val_cross_inputs, val_deep_input, val_wide_input, val_inputs_slice, val_labels
    gc.collect()

    print('*-' * 40, 'LOAD TEST DATA')
    making_data_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/order_xt/'
    link_data_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/max_170_link_sqe_for_order/'
    cross_data_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/cross_sqe_for_order/'
    head_link_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/head_link_data_clear/'
    win_order_test_data_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/win_order_xw/'
    pre_arrival_sqe_test_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/sqe_arrival_for_link/'
    data_test_for_driver_xw = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/data_for_driver_xw/'
    downstream_status_test_dir = '/home/didi2021/didi2021/giscup_2021/final_test_data_0703/downstream_status_for_order/'
    test_data, _, _, _ = process.load_data(making_data_dir,
                                                                            link_data_dir,
                                                                            cross_data_dir,
                                                                            head_link_dir,
                                                                            win_order_test_data_dir,
                                                                            pre_arrival_sqe_test_dir,
                                                                            data_test_for_driver_xw,
                                                                            downstream_status_test_dir)

    # PROCESSING DATA
    print('*-' * 40, 'PROCESSING DATA')
    test_data = process.processing_data(test_data, mk_cols_list, link_cols_list, cross_cols_list,
                                                              WIDE_COLS, is_test=True)
    print('*-' * 40, 'PROCESSING INPUTS FOR TEST_DATA', test_data.shape)
    test_link_inputs, test_cross_inputs, test_deep_input, test_wide_input, \
    test_inputs_slice, test_labels = process.processing_inputs(
        test_data, mk_cols_list, link_cols_list, cross_cols_list, WIDE_COLS)
    test_pre = test_data[['order_id']]
    del test_data
    gc.collect()

    # MODEL_RPEDICT
    print('*-' * 40, 'MODEL_RPEDICT')
    test_pre = test_pre.rename(columns={'order_id': 'id'})
    test_pred = model.predict(
        [test_link_inputs, test_cross_inputs, test_deep_input, test_wide_input, test_inputs_slice],
        batch_size=2048)
    test_pre['test_predict'] = test_pred
    # test_pre['test_predict'] = test_pre['test_predict'].round(0)
    test_pre = test_pre.rename(columns={'test_predict': 'result'})  # 更改列名
    test_pre = test_pre[['id', 'result']]
    print(test_pre.head())
    result_save_path = '../result_csv/submit_w_0730_5.csv'
    print('*-' * 40, 'CSV_SAVE_PATH:', result_save_path)
    test_pre.to_csv(result_save_path, index=0)  # 保存
    print('..........Finish')
    t2 = time.time()
    print("Total time spent: {:.4f}".format((t2-t1)/3600))
