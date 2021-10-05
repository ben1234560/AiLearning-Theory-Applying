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
#from keras_radam import RAdam
from keras_radam.training import RAdamOptimizer
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, Conv1D
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
import os
from tensorflow.keras.losses import mean_absolute_percentage_error
#from tensorflow.contrib.opt import AdamWOptimizer


os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
gamma = 2.0
alpha=.25
epsilon = K.epsilon()


def mape_2(y_true, y_pred):
    y_true = y_true[:, :1]
    y_pred = y_pred[:, :1]
    return tf.py_function(mean_absolute_percentage_error, (y_true, y_pred), tf.float32)

def mape_3(y_true, y_pred):
    y_true = y_true[:, :1]
    y_pred = y_pred[:, 1:]
    return tf.py_function(mean_absolute_percentage_error, (y_true, y_pred), tf.float32)


def knowledge_distillation_loss_withFL(y_true, y_pred, beta=0.1):

    # Extract the groundtruth from dataset and the prediction from teacher model
    y_true, y_pred_teacher = y_true[: , :1], y_true[: , 1:]
    
    # Extract the prediction from student model
    y_pred, y_pred_stu = y_pred[: , :1], y_pred[: , 1:]

    loss = beta*focal_loss(y_true,y_pred) + (1-beta)*mean_absolute_percentage_error(y_pred_teacher, y_pred_stu)

    return loss


def focal_loss(y_true, y_pred):
    pt_1 = y_pred * y_true
    pt_1 = K.clip(pt_1, epsilon, 1-epsilon)
    CE_1 = -K.log(pt_1)
    FL_1 = alpha* K.pow(1-pt_1, gamma) * CE_1
    
    pt_0 = (1-y_pred) * (1-y_true)
    pt_0 = K.clip(pt_0, epsilon, 1-epsilon)
    CE_0 = -K.log(pt_0)
    FL_0 = (1-alpha)* K.pow(1-pt_0, gamma) * CE_0
    
    loss = K.sum(FL_1, axis=1) + K.sum(FL_0, axis=1)
    return loss


def knowledge_distillation_loss_withBE(y_true, y_pred, beta=0.6):

    # Extract the groundtruth from dataset and the prediction from teacher model
    y_true, y_pred_teacher = y_true[: , :1], y_true[: , 1:]
    
    # Extract the prediction from student model
    y_pred, y_pred_stu = y_pred[: , :1], y_pred[: , 1:]

    loss = beta*mean_absolute_percentage_error(y_true,y_pred) + (1-beta)*mean_absolute_percentage_error(y_pred_teacher, y_pred_stu)

    return loss


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(
        hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))


def lstm_layer(hidden_dim, dropout):
    return L.Bidirectional(L.LSTM(
        hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))


class FeatureDictionary(object):
    def __init__(self, df=None, numeric_cols=[], ignore_cols=[], cate_cols=[]):
        self.df = df
        self.cate_cols = cate_cols
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()  # feat_dict 获取cate feature每一列的字典长度。

    def gen_feat_dict(self):
        self.feat_cate_len = {}
        tc = 0
        for col in self.cate_cols:
            # 获取每一列的类别
            us = self.df[col].unique()
            us_len = len(us)
            # 获取每一列的类别对应的维度
            self.feat_cate_len[col] = us_len


def embedding_layers(fd):
    # 该函数主要是定义输入和embedding输入的网络层
    embeddings_tensors = []
    continus_tensors = []
    cate_feature = fd.feat_cate_len
    numeric_feature = fd.numeric_cols
    for ec in cate_feature:
        layer_name = ec + '_inp'
        # for categorical features, embedding特征在维度保持在6×(category cardinality)**(1/4)
        embed_dim = cate_feature[ec] if int(6 * np.power(cate_feature[ec], 1 / 4)) > cate_feature[ec] else int(
            6 * np.power(cate_feature[ec], 1 / 4))
        t_inp, t_embedding = embedding_input(layer_name, cate_feature[ec], embed_dim)
        embeddings_tensors.append((t_inp, t_embedding))
        del (t_inp, t_embedding)
    for cc in numeric_feature:
        layer_name = cc + '_in'
        t_inp, t_build = continus_input(layer_name)
        continus_tensors.append((t_inp, t_build))
        del (t_inp, t_build)
    # category feature的输入 这里的输入特征顺序要与xu
    inp_layer = [et[0] for et in embeddings_tensors]
    inp_embed = [et[1] for et in embeddings_tensors]
    # numeric feature的输入
    inp_layer += [ct[0] for ct in continus_tensors]
    inp_embed += [ct[1] for ct in continus_tensors]

    return inp_layer, inp_embed


def embedding_input(name, input_dim, output_dim):
    inp = L.Input(shape=(1,), dtype='int64', name=name)
    embeddings = L.Embedding(input_dim, output_dim, input_length=1)(inp)
    return inp, embeddings


def continus_input(name):
    inp = L.Input(shape=(1,), dtype='float32', name=name)
    return inp, L.Reshape((1, 1))(inp)


class CrossLayer(L.Layer):
    def __init__(self, output_dim, num_layer, **kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        super(CrossLayer, self).__init__(**kwargs)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
            'num_layers': self.num_layers,
            'units': self.units,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
        })
        return config

    def build(self, input_shape):
        self.input_dim = input_shape[2]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(
                self.add_weight(shape=[1, self.input_dim], initializer='glorot_uniform', name='w_{}'.format(i),
                                trainable=True))
            self.bias.append(
                self.add_weight(shape=[1, self.input_dim], initializer='zeros', name='b_{}'.format(i), trainable=True))
        self.built = True

    def call(self, input):
        for i in range(self.num_layer):
            if i == 0:
                cross = L.Lambda(lambda x: K.batch_dot(K.dot(x, K.transpose(self.W[i])), x) + self.bias[i] + x)(input)
            else:
                cross = L.Lambda(lambda x: K.batch_dot(K.dot(x, K.transpose(self.W[i])), input) + self.bias[i] + x)(
                    cross)
        return L.Flatten()(cross)

    def compute_output_shape(self, input_shape):
        return None, self.output_dim


def preprocess(df, cate_cols, numeric_cols):
    for cl in cate_cols:
        le = LabelEncoder()
        df[cl] = le.fit_transform(df[cl])
    cols = cate_cols + numeric_cols
    X_train = df[cols]
    return X_train


def DCN_model(inp_layer, inp_embed, link_size, cross_size, slice_size, input_deep_col, input_wide_col,
              link_nf_size, cross_nf_size, encoder,  link_seqlen=170, cross_seqlen=12, pred_len=1,
              dropout=0.25, sp_dropout=0.1, embed_dim=64, hidden_dim=128, n_layers=3, lr=0.001, 
              kernel_size1=3, kernel_size2=2, conv_size=128, conv=False, have_knowledge=True):
    inp = L.concatenate(inp_embed, axis=-1)
    link_inputs = L.Input(shape=(link_seqlen, link_nf_size), name='link_inputs')
    cross_inputs = L.Input(shape=(cross_seqlen, cross_nf_size), name='cross_inputs')
    deep_inputs = L.Input(shape=(input_deep_col,), name='deep_input')
    slice_input = L.Input(shape=(1,), name='slice_input')
    wide_inputs = keras.layers.Input(shape=(input_wide_col,), name='wide_inputs')

    # link----------------------------
    categorical_link = link_inputs[:, :, :1]
    embed_link = L.Embedding(input_dim=link_size, output_dim=embed_dim, mask_zero=True)(categorical_link)
    reshaped_link = tf.reshape(embed_link, shape=(-1, embed_link.shape[1], embed_link.shape[2] * embed_link.shape[3]))
    reshaped_link = L.SpatialDropout1D(sp_dropout)(reshaped_link)
    
    """
    categorical_slice = link_inputs[:, :, 5:6]
    embed_slice = L.Embedding(input_dim=289, output_dim=16, mask_zero=True)(categorical_slice)
    reshaped_slice = tf.reshape(embed_slice, shape=(-1, embed_slice.shape[1], embed_slice.shape[2] * embed_slice.shape[3]))
    reshaped_slice = L.SpatialDropout1D(sp_dropout)(reshaped_slice)

    categorical_hightemp = link_inputs[:, :, 6:7]
    embed_hightemp = L.Embedding(input_dim=33, output_dim=8, mask_zero=True)(categorical_hightemp)
    reshaped_hightemp = tf.reshape(embed_hightemp, shape=(-1, embed_hightemp.shape[1], embed_hightemp.shape[2] * embed_hightemp.shape[3]))
    reshaped_hightemp = L.SpatialDropout1D(sp_dropout)(reshaped_hightemp)

    categorical_weather = link_inputs[:, :, 7:8]
    embed_weather = L.Embedding(input_dim=7, output_dim=8, mask_zero=True)(categorical_weather)
    reshaped_weather = tf.reshape(embed_weather, shape=(-1, embed_weather.shape[1], embed_weather.shape[2] * embed_weather.shape[3]))
    reshaped_weather = L.SpatialDropout1D(sp_dropout)(reshaped_weather)
    
    numerical_fea1 = link_inputs[:, :, 1:5]
    numerical_fea1 = L.Masking(mask_value=0, name='numerical_fea1')(numerical_fea1)
    hidden = L.concatenate([reshaped_link, numerical_fea1, reshaped_slice, reshaped_hightemp, reshaped_weather], axis=2)
    
    """
    if have_knowledge:
        numerical_fea1 = link_inputs[:, :, 1:5]
        numerical_fea1 = L.Masking(mask_value=0, name='numerical_fea1')(numerical_fea1)
       
         
        categorical_ar_st = link_inputs[:, :, 5:6]
        categorical_ar_st = L.Masking(mask_value=-1, name='categorical_ar_st')(categorical_ar_st)
        embed_ar_st = L.Embedding(input_dim=289, output_dim=8)(categorical_ar_st)
        reshaped_ar_st = tf.reshape(embed_ar_st, shape=(-1, embed_ar_st.shape[1], embed_ar_st.shape[2] * embed_ar_st.shape[3]))
        reshaped_ar_st = L.SpatialDropout1D(sp_dropout)(reshaped_ar_st)

        categorical_ar_sl = link_inputs[:, :, 6:7]
        categorical_ar_sl = L.Masking(mask_value=-1, name='categorical_ar_sl')(categorical_ar_sl)
        embed_ar_sl = L.Embedding(input_dim=289, output_dim=8)(categorical_ar_sl)
        reshaped_ar_sl = tf.reshape(embed_ar_sl, shape=(-1, embed_ar_sl.shape[1], embed_ar_sl.shape[2] * embed_ar_sl.shape[3]))
        reshaped_ar_sl = L.SpatialDropout1D(sp_dropout)(reshaped_ar_sl)
        hidden = L.concatenate([reshaped_link, reshaped_ar_st, reshaped_ar_sl, numerical_fea1],axis=2)
        
        #hidden = L.concatenate([reshaped_link, numerical_fea1],axis=2)
    else:
        numerical_fea1 = link_inputs[:, :, 1:5]
        numerical_fea1 = L.Masking(mask_value=0, name='numerical_fea1')(numerical_fea1)    
        
        categorical_arrival = link_inputs[:, :, 5:6]
        categorical_arrival = L.Masking(mask_value=-1, name='categorical_arrival')(categorical_arrival)
        embed_ar = L.Embedding(input_dim=5, output_dim=16)(categorical_arrival)
        reshaped_ar = tf.reshape(embed_ar, shape=(-1, embed_ar.shape[1], embed_ar.shape[2] * embed_ar.shape[3]))
        reshaped_ar = L.SpatialDropout1D(sp_dropout)(reshaped_ar)
        
        categorical_ar_st = link_inputs[:, :, 6:7]
        categorical_ar_st = L.Masking(mask_value=-1, name='categorical_ar_st')(categorical_ar_st)
        embed_ar_st = L.Embedding(input_dim=289, output_dim=8)(categorical_ar_st)
        reshaped_ar_st = tf.reshape(embed_ar_st, shape=(-1, embed_ar_st.shape[1], embed_ar_st.shape[2] * embed_ar_st.shape[3]))
        reshaped_ar_st = L.SpatialDropout1D(sp_dropout)(reshaped_ar_st)

        categorical_ar_sl = link_inputs[:, :, 7:8]
        categorical_ar_sl = L.Masking(mask_value=-1, name='categorical_ar_sl')(categorical_ar_sl)
        embed_ar_sl = L.Embedding(input_dim=289, output_dim=8)(categorical_ar_sl)
        reshaped_ar_sl = tf.reshape(embed_ar_sl, shape=(-1, embed_ar_sl.shape[1], embed_ar_sl.shape[2] * embed_ar_sl.shape[3]))
        reshaped_ar_sl = L.SpatialDropout1D(sp_dropout)(reshaped_ar_sl)
        hidden = L.concatenate([reshaped_link, reshaped_ar, reshaped_ar_st, reshaped_ar_sl, numerical_fea1],axis=2)
        
        #hidden = L.concatenate([reshaped_link, reshaped_ar, numerical_fea1],axis=2)
    #hidden = L.Masking(mask_value=0)(hidden)
    for x in range(n_layers):
        hidden = gru_layer(hidden_dim, dropout)(hidden)

    if conv:
        x_conv1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(hidden)
        avg_pool1_gru = GlobalAveragePooling1D()(x_conv1)
        max_pool1_gru = GlobalMaxPooling1D()(x_conv1)
        #x_conv2 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(hidden)
        #avg_pool2_gru = GlobalAveragePooling1D()(x_conv2)
        #max_pool2_gru = GlobalMaxPooling1D()(x_conv2)
        truncated_link = concatenate([avg_pool1_gru, max_pool1_gru])
    else:
        truncated_link = hidden[:, :pred_len]
        truncated_link = L.Flatten()(truncated_link)

    # truncated_link = Attention(256)(hidden)
    # CROSS----------------------------
    categorical_fea2 = cross_inputs[:, :, :1]
    embed2 = L.Embedding(input_dim=cross_size, output_dim=16, mask_zero=True)(categorical_fea2)
    reshaped2 = tf.reshape(embed2, shape=(-1, embed2.shape[1], embed2.shape[2] * embed2.shape[3]))
    reshaped2 = L.SpatialDropout1D(sp_dropout)(reshaped2)

    numerical_fea2 = cross_inputs[:, :, 1:]
    numerical_fea2 = L.Masking(mask_value=0, name='numerical_fea2')(numerical_fea2)
    hidden2 = L.concatenate([reshaped2, numerical_fea2], axis=2)
    # hidden2 = L.Masking(mask_value=0)(hidden2)
    for x in range(n_layers):
        hidden2 = gru_layer(hidden_dim, dropout)(hidden2)

    if conv:
        x_conv3 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(hidden2)
        avg_pool3_gru = GlobalAveragePooling1D()(x_conv3)
        max_pool3_gru = GlobalMaxPooling1D()(x_conv3)
        #x_conv4 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(hidden2)
        #avg_pool4_gru = GlobalAveragePooling1D()(x_conv4)
        #max_pool4_gru = GlobalMaxPooling1D()(x_conv4)
        truncated_cross = concatenate([avg_pool3_gru, max_pool3_gru])
    else:
        truncated_cross = hidden2[:, :pred_len]
        truncated_cross = L.Flatten()(truncated_cross)
    
    # truncated_cross = Attention(256)(hidden2)
    # SLICE----------------------------
    embed_slice = L.Embedding(input_dim=slice_size, output_dim=1)(slice_input)
    embed_slice = L.Flatten()(embed_slice)

    # DEEP_INPUS
    x = encoder(deep_inputs)
    x = L.Concatenate()([x, deep_inputs])  # use both raw and encoded features
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.25)(x)
    
    for i in range(3):
        x = L.Dense(256)(x)
        x = L.BatchNormalization()(x)
        x = L.Lambda(tf.keras.activations.swish)(x)
        x = L.Dropout(0.25)(x)
    dense_hidden3 = L.Dense(64,activation='linear')(x)

    # DCN
    cross = CrossLayer(output_dim=inp.shape[2], num_layer=8, name="cross_layer")(inp)


    # MAIN-------------------------------
    truncated = L.concatenate([truncated_link, truncated_cross, cross, dense_hidden3, wide_inputs, embed_slice])
    truncated = L.BatchNormalization()(truncated)
    truncated = L.Dropout(dropout)(L.Dense(512, activation='relu') (truncated))
    truncated = L.BatchNormalization()(truncated)
    truncated = L.Dropout(dropout)(L.Dense(256, activation='relu') (truncated))

    if have_knowledge:
        out = L.Dense(2, activation='linear', name='out')(truncated)
        model = tf.keras.Model(inputs=[inp_layer, link_inputs, cross_inputs, deep_inputs, wide_inputs, slice_input],
                               outputs=out)
        print(model.summary())
        model.compile(loss=knowledge_distillation_loss_withBE,
                      optimizer=RAdamOptimizer(learning_rate=1e-3),  # 'adam'  RAdam(warmup_proportion=0.1, min_lr=1e-7)
                      #metrics={'out':'mape'} # AdamWOptimizer(weight_decay=1e-4)
                      metrics=[mape_2,mape_3]
                      )
    else:
        out = L.Dense(1, activation='linear', name='out')(truncated)
        model = tf.keras.Model(inputs=[inp_layer, link_inputs, cross_inputs, deep_inputs, wide_inputs, slice_input],
                               outputs=out)
        print(model.summary())
        model.compile(loss=['mape'],
                      optimizer=RAdamOptimizer(learning_rate=1e-3),  # 'adam'  RAdam(warmup_proportion=0.1, min_lr=1e-7)
                      #metrics={'out':'mape'}
                      metrics=['mape']
                      )

    return model


def arrival_model(inp_layer, inp_embed, link_size, cross_size, slice_size, input_deep_col, input_wide_col,
              link_nf_size, cross_nf_size,  link_seqlen=170, cross_seqlen=12, pred_len=1,
              dropout=0.25, sp_dropout=0.1, embed_dim=64, hidden_dim=128, n_layers=3, lr=0.001,
              kernel_size1=3, kernel_size2=2, conv_size=128, conv=False):
    inp = L.concatenate(inp_embed, axis=-1)
    link_inputs = L.Input(shape=(link_seqlen, link_nf_size), name='link_inputs')
    cross_inputs = L.Input(shape=(cross_seqlen, cross_nf_size), name='cross_inputs')
    deep_inputs = L.Input(shape=(input_deep_col,), name='deep_input')
    slice_input = L.Input(shape=(1,), name='slice_input')
    wide_inputs = keras.layers.Input(shape=(input_wide_col,), name='wide_inputs')

    # link----------------------------
    categorical_link = link_inputs[:, :, :1]
    embed_link = L.Embedding(input_dim=link_size, output_dim=embed_dim, mask_zero=True)(categorical_link)
    reshaped_link = tf.reshape(embed_link, shape=(-1, embed_link.shape[1], embed_link.shape[2] * embed_link.shape[3]))
    reshaped_link = L.SpatialDropout1D(sp_dropout)(reshaped_link)
    """ 
    categorical_slice = link_inputs[:, :, 5:6]
    embed_slice = L.Embedding(input_dim=289, output_dim=16, mask_zero=True)(categorical_slice)
    reshaped_slice = tf.reshape(embed_slice, shape=(-1, embed_slice.shape[1], embed_slice.shape[2] * embed_slice.shape[3]))
    reshaped_slice = L.SpatialDropout1D(sp_dropout)(reshaped_slice)

    categorical_hightemp = link_inputs[:, :, 6:7]
    embed_hightemp = L.Embedding(input_dim=33, output_dim=8, mask_zero=True)(categorical_hightemp)
    reshaped_hightemp = tf.reshape(embed_hightemp, shape=(-1, embed_hightemp.shape[1], embed_hightemp.shape[2] * embed_hightemp.shape[3]))
    reshaped_hightemp = L.SpatialDropout1D(sp_dropout)(reshaped_hightemp)

    categorical_weather = link_inputs[:, :, 7:8]
    embed_weather = L.Embedding(input_dim=7, output_dim=8, mask_zero=True)(categorical_weather)
    reshaped_weather = tf.reshape(embed_weather, shape=(-1, embed_weather.shape[1], embed_weather.shape[2] * embed_weather.shape[3]))
    reshaped_weather = L.SpatialDropout1D(sp_dropout)(reshaped_weather)
    
    numerical_fea1 = link_inputs[:, :, 1:5]
    numerical_fea1 = L.Masking(mask_value=0, name='numerical_fea1')(numerical_fea1)
    hidden = L.concatenate([reshaped_link, numerical_fea1, reshaped_slice, reshaped_hightemp, reshaped_weather], axis=2)
    """
    numerical_fea1 = link_inputs[:, :, 1:]
    numerical_fea1 = L.Masking(mask_value=0, name='numerical_fea1')(numerical_fea1)
    hidden = L.concatenate([reshaped_link, numerical_fea1],axis=2)
    
    #hidden = L.Masking(mask_value=0)(hidden)
    for x in range(n_layers):
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    if conv:
        x_conv1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(hidden)
        avg_pool1_gru = GlobalAveragePooling1D()(x_conv1)
        max_pool1_gru = GlobalMaxPooling1D()(x_conv1)
        #x_conv2 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(hidden)
        #avg_pool2_gru = GlobalAveragePooling1D()(x_conv2)
        #max_pool2_gru = GlobalMaxPooling1D()(x_conv2)
        truncated_link = concatenate([avg_pool1_gru, max_pool1_gru])
    else:
        truncated_link = hidden[:, :pred_len]
        truncated_link = L.Flatten()(truncated_link)

    # truncated_link = Attention(256)(hidden)
    # CROSS----------------------------
    categorical_fea2 = cross_inputs[:, :, :1]
    embed2 = L.Embedding(input_dim=cross_size, output_dim=16, mask_zero=True)(categorical_fea2)
    reshaped2 = tf.reshape(embed2, shape=(-1, embed2.shape[1], embed2.shape[2] * embed2.shape[3]))
    reshaped2 = L.SpatialDropout1D(sp_dropout)(reshaped2)

    numerical_fea2 = cross_inputs[:, :, 1:]
    numerical_fea2 = L.Masking(mask_value=0, name='numerical_fea2')(numerical_fea2)
    hidden2 = L.concatenate([reshaped2, numerical_fea2], axis=2)
    # hidden2 = L.Masking(mask_value=0)(hidden2)
    for x in range(n_layers):
        hidden2 = gru_layer(hidden_dim, dropout)(hidden2)

    if conv:
        x_conv3 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(hidden2)
        avg_pool3_gru = GlobalAveragePooling1D()(x_conv3)
        max_pool3_gru = GlobalMaxPooling1D()(x_conv3)
        #x_conv4 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(hidden2)
        #avg_pool4_gru = GlobalAveragePooling1D()(x_conv4)
        #max_pool4_gru = GlobalMaxPooling1D()(x_conv4)
        truncated_cross = concatenate([avg_pool3_gru, max_pool3_gru])
    else:
        truncated_cross = hidden2[:, :pred_len]
        truncated_cross = L.Flatten()(truncated_cross)

    # truncated_cross = Attention(256)(hidden2)
    # SLICE----------------------------
    embed_slice = L.Embedding(input_dim=slice_size, output_dim=1)(slice_input)
    embed_slice = L.Flatten()(embed_slice)

    # DEEP_INPUS
    x = L.BatchNormalization()(deep_inputs)
    x = L.Dropout(0.25)(x)

    for i in range(3):
        x = L.Dense(256)(x)
        x = L.BatchNormalization()(x)
        x = L.Lambda(tf.keras.activations.swish)(x)
        x = L.Dropout(0.25)(x)
    dense_hidden3 = L.Dense(64,activation='linear')(x)

    # DCN
    cross = CrossLayer(output_dim=inp.shape[2], num_layer=8, name="cross_layer")(inp)
    truncated = L.concatenate([truncated_link, truncated_cross, cross, dense_hidden3, wide_inputs, embed_slice])
    truncated = L.BatchNormalization()(truncated)
    truncated = L.Dropout(dropout)(L.Dense(512, activation='relu') (truncated))
    truncated = L.BatchNormalization()(truncated)
    truncated = L.Dropout(dropout)(L.Dense(256, activation='relu') (truncated))

    arrival_0 = L.Dense(1, activation='linear', name='arrival_0')(truncated)
    arrival_1 = L.Dense(1, activation='linear', name='arrival_1')(truncated)
    arrival_2 = L.Dense(1, activation='linear', name='arrival_2')(truncated)
    arrival_3 = L.Dense(1, activation='linear', name='arrival_3')(truncated)
    arrival_4 = L.Dense(1, activation='linear', name='arrival_4')(truncated)

    model = tf.keras.Model(inputs=[inp_layer,link_inputs, cross_inputs, deep_inputs, wide_inputs, slice_input],
                           outputs=[arrival_0,arrival_1,arrival_2,arrival_3,arrival_4])
    print(model.summary())
    model.compile(loss='mse',
                  optimizer=RAdamOptimizer(learning_rate=1e-3)  # 'adam'  RAdam(warmup_proportion=0.1, min_lr=1e-7)
                  )
                  
    return model


def get_mc_es_lr(model_name: str, patience=5, min_delta=1e-4):
    mc = tf.keras.callbacks.ModelCheckpoint('../model_h5/model_{}.h5'.format(model_name)),
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                          restore_best_weights=True, patience=patience)
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=patience-1, mode='min',
                                              min_delta=min_delta)

    return mc, es, lr


def get_mc_es_lr_for_student(model_name: str, patience=5, min_delta=1e-4):
    mc = tf.keras.callbacks.ModelCheckpoint('../model_h5/model_{}.h5'.format(model_name)),
    es = tf.keras.callbacks.EarlyStopping(monitor='val_mape_2', mode='min',
                                          restore_best_weights=True, patience=patience)
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mape_2', factor=0.8, patience=patience, mode='min',
                                              min_delta=min_delta)

    return mc, es, lr



def create_autoencoder(input_dim, output_dim, noise=0.05):
    i = L.Input(input_dim)
    encoded = L.BatchNormalization()(i)
    encoded = L.GaussianNoise(noise)(encoded)
    encoded = L.Dense(128, activation='relu')(encoded)
    decoded = L.Dropout(0.2)(encoded)
    decoded = L.Dense(input_dim,name='decoded')(decoded)
    x = L.Dense(64, activation='relu')(decoded)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.2)(x)
    x = L.Dense(output_dim, activation='linear', name='ata_output')(x)
    
    encoder = keras.models.Model(inputs=i, outputs=decoded)
    autoencoder = keras.models.Model(inputs=i, outputs=[decoded, x])
    
    autoencoder.compile(optimizer=RAdamOptimizer(learning_rate=1e-3), loss={'decoded':'mse', 'ata_output': 'mape'})
    return autoencoder, encoder


class Attention(L.Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
            loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)


