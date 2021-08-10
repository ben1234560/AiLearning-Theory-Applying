import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow.keras.layers as L
# import tensorflow.keras.models as M
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras_radam.training import RAdamOptimizer
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, Conv1D
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(
        hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))


def lstm_layer(hidden_dim, dropout):
    return L.Bidirectional(L.LSTM(
        hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))


def preprocess(df, cate_cols, numeric_cols):
    for cl in cate_cols:
        le = LabelEncoder()
        df[cl] = le.fit_transform(df[cl])
    cols = cate_cols + numeric_cols
    X_train = df[cols]
    return X_train


def wd_model(link_size, cross_size, slice_size, input_deep_col, input_wide_col,
              link_nf_size, cross_nf_size, link_seqlen=170, cross_seqlen=12, pred_len=1,
              dropout=0.25, sp_dropout=0.1, embed_dim=64, hidden_dim=128, n_layers=3, lr=0.001,
              kernel_size1=3, kernel_size2=2, conv_size=128, conv='conv'):
    link_inputs = L.Input(shape=(link_seqlen, link_nf_size))
    cross_inputs = L.Input(shape=(cross_seqlen, cross_nf_size))
    deep_inputs = L.Input(shape=(input_deep_col,), name='deep_input')
    slice_input = L.Input(shape=(1,))
    wide_inputs = keras.layers.Input(shape=(input_wide_col,), name='wide_input')

    # link----------------------------
    categorical_fea1 = link_inputs[:, :, :1]
    numerical_fea1 = link_inputs[:, :, 1:5]

    embed = L.Embedding(input_dim=link_size, output_dim=embed_dim)(categorical_fea1)
    reshaped = tf.reshape(embed, shape=(-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
    #reshaped = L.SpatialDropout1D(sp_dropout)(reshaped)

    hidden = L.concatenate([reshaped, numerical_fea1], axis=2)
    hidden = L.SpatialDropout1D(sp_dropout)(hidden)
    """
    categorical_ar_st = link_inputs[:, :, 5:6]
    categorical_ar_st = L.Masking(mask_value=-1, name='categorical_ar_st')(categorical_ar_st)
    embed_ar_st = L.Embedding(input_dim=(-1,289), output_dim=8)(categorical_ar_st)
    reshaped_ar_st = tf.reshape(embed_ar_st, shape=(-1, embed_ar_st.shape[1], embed_ar_st.shape[2] * embed_ar_st.shape[3]))
    reshaped_ar_st = L.SpatialDropout1D(sp_dropout)(reshaped_ar_st)

    categorical_ar_sl = link_inputs[:, :, 6:7]
    categorical_ar_sl = L.Masking(mask_value=-1, name='categorical_ar_sl')(categorical_ar_sl)
    embed_ar_sl = L.Embedding(input_dim=(-1, 289), output_dim=8)(categorical_ar_sl)
    reshaped_ar_sl = tf.reshape(embed_ar_sl, shape=(-1, embed_ar_sl.shape[1], embed_ar_sl.shape[2] * embed_ar_sl.shape[3]))
    reshaped_ar_sl = L.SpatialDropout1D(sp_dropout)(reshaped_ar_sl)
    hidden = L.concatenate([reshaped, reshaped_ar_st, reshaped_ar_sl, numerical_fea1],axis=2)
    """
    for x in range(n_layers):
        hidden = lstm_layer(hidden_dim, dropout)(hidden)

    if conv=='conv':
        #x_conv1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(hidden)
        avg_pool1_gru = GlobalAveragePooling1D()(hidden)
        max_pool1_gru = GlobalMaxPooling1D()(hidden)
        truncated_link = concatenate([avg_pool1_gru, max_pool1_gru])
    elif conv=='resnet50':
        truncated_link = ResNet50(include_top=False, pooling='max', weights=None)(hidden)
    else:
        truncated_link = hidden[:, :pred_len]
        truncated_link = L.Flatten()(truncated_link)

    # cross----------------------------
    categorical_fea2 = cross_inputs[:, :, :1]
    numerical_fea2 = cross_inputs[:, :, 1:]
    embed2 = L.Embedding(input_dim=cross_size, output_dim=embed_dim)(categorical_fea2)
    reshaped2 = tf.reshape(embed2, shape=(-1, embed2.shape[1], embed2.shape[2] * embed2.shape[3]))
    #reshaped2 = L.SpatialDropout1D(sp_dropout)(reshaped2)

    hidden2 = L.concatenate([reshaped2, numerical_fea2], axis=2)
    hidden2 = L.SpatialDropout1D(sp_dropout)(hidden2)
    for x in range(n_layers):
        hidden2 = lstm_layer(hidden_dim, dropout)(hidden2)

    if conv=='conv':
        #x_conv3 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(hidden2)
        avg_pool3_gru = GlobalAveragePooling1D()(hidden2)
        max_pool3_gru = GlobalMaxPooling1D()(hidden2)
        truncated_cross = concatenate([avg_pool3_gru, max_pool3_gru])
    elif conv=='resnet50':
        truncated_cross = ResNet50(include_top=False, pooling='max', weights=None)(hidden2)
    else:
        truncated_cross = hidden2[:, :pred_len]
        truncated_cross = L.Flatten()(truncated_cross)

    # slice----------------------------
    embed_slice = L.Embedding(input_dim=slice_size, output_dim=1)(slice_input)
    embed_slice = L.Flatten()(embed_slice)

    # deep_inputs
    """
    dense_hidden1 = L.Dense(256, activation="relu")(deep_inputs)
    dense_hidden1 = L.Dropout(dropout)(dense_hidden1)
    dense_hidden2 = L.Dense(256, activation="relu")(dense_hidden1)
    dense_hidden2 = L.Dropout(dropout)(dense_hidden2)
    dense_hidden3 = L.Dense(128, activation="relu")(dense_hidden2)
    """
    x = L.Dense(512, activation="relu")(deep_inputs)
    x = L.BatchNormalization()(x)
    x = L.Lambda(tf.keras.activations.swish)(x)
    x = L.Dropout(0.25)(x)
    for i in range(2):
        x = L.Dense(256)(x)
        x = L.BatchNormalization()(x)
        x = L.Lambda(tf.keras.activations.swish)(x)
        x = L.Dropout(0.25)(x)
    dense_hidden3 = L.Dense(64,activation='linear')(x)
    # main-------------------------------
    truncated = L.concatenate([truncated_link, truncated_cross, dense_hidden3, wide_inputs, embed_slice])  # WD
    """
    truncated = L.BatchNormalization()(truncated)
    truncated = L.Dropout(dropout)(L.Dense(512, activation='relu') (truncated))
    truncated = L.BatchNormalization()(truncated)
    truncated = L.Dropout(dropout)(L.Dense(256, activation='relu') (truncated))
    """
    truncated = L.BatchNormalization()(truncated)
    truncated = L.Dropout(dropout)(L.Dense(1024, activation='relu') (truncated))
    truncated = L.Dropout(dropout)(truncated)

    for i in range(2):
        truncated = L.Dense(512)(truncated)
        truncated = L.BatchNormalization()(truncated)
        truncated = L.Lambda(tf.keras.activations.swish)(truncated)
        truncated = L.Dropout(dropout)(truncated)

    out = L.Dense(1, activation='linear')(truncated)


    model = tf.keras.Model(inputs=[link_inputs, cross_inputs, deep_inputs, wide_inputs, slice_input],
                           outputs=out)  # WD
    print(model.summary())
    model.compile(loss='mape',
                  optimizer=RAdamOptimizer(learning_rate=1e-3),
                  metrics=['mape'])

    return model


def get_mc_es_lr(model_name: str, patience=5, min_delta=1e-4):
    mc = tf.keras.callbacks.ModelCheckpoint('../model_h5/model_{}.h5'.format(model_name)),
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                          restore_best_weights=True, patience=patience)
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=patience, mode='min',
                                              min_delta=min_delta)

    return mc, es, lr


class Mish(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def mish(x):
        return tf.keras.layers.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)


tf.keras.utils.get_custom_objects().update({'mish': tf.keras.layers.Activation(mish)})
