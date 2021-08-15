#coding=utf-8
"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2021.08.01
import numpy as np
import networkx as nx
import pandas as pd
from gem.embedding.node2vec import node2vec
import os
from utils import parallel_apply
from functools import partial
import gc
def link_id_find(gr):
    gr_ = gr.copy()
    tmp = list(gr_['link_id'])
    link_id_tuple = []
    for i in range(len(tmp)-1):
        link_id_tuple.append([tmp[i],tmp[i+1]])
    return link_id_tuple

if __name__ == '__main__':
    root_path = '../data/giscup_2021/'
    nrows = None
    ######################################nextlinks #######################################
    nextlinks = pd.read_csv(root_path + 'nextlinks.txt', sep=' ', header=None)
    nextlinks.columns = ['from_id', 'to_id']
    nextlinks['to_id'] = nextlinks['to_id'].astype('str')
    nextlinks['to_id'] = nextlinks['to_id'].apply(lambda x: x.split(","))
    nextlinks = pd.DataFrame({'from_id': nextlinks.from_id.repeat(nextlinks.to_id.str.len()),
                              'to_id': np.concatenate(nextlinks.to_id.values)})
    nextlinks['from_id'] = nextlinks['from_id'].astype(int)
    nextlinks['to_id'] = nextlinks['to_id'].astype(int)
    from_id = nextlinks['from_id'].unique()
    # nextlinks.to_csv('../data/giscup_2021/nextlink_all.csv',index=False)
    # nextlinks = pd.read_csv('../data/giscup_2021/nextlink_all.csv')

    ######################################nextlinks #######################################
    if 'nextlinks_allday.csv' in os.listdir(root_path):
        nextlinks = pd.read_csv(root_path + 'nextlinks_allday.csv')
    else:
        nextlinks_new = []
        for name in os.listdir(root_path + 'train/'):
            data_time = name.split('.')[0]
            if data_time == '20200803':
                continue
            train = pd.read_csv(root_path + 'train/{}'.format(name),sep= ';;',header=None,nrows=nrows)
            train_head = pd.DataFrame(train[0].str.split(' ').tolist(),
                                      columns=['order_id', 'ata', 'distance', 'simple_eta', 'driver_id', 'slice_id'])
            train_head['order_id'] = train_head['order_id'].astype(str)
            train_head['ata'] = train_head['ata'].astype(float)
            train_head['distance'] = train_head['distance'].astype(float)
            train_head['simple_eta'] = train_head['simple_eta'].astype(float)
            train_head['driver_id'] = train_head['driver_id'].astype(int)
            train_head['slice_id'] = train_head['slice_id'].astype(int)
            data_link = train[[1]]
            print("flag:", 1)
            data_link['index'] = train_head.index
            data_link['order_id'] = train_head['order_id']
            print("flag:", 2)
            data_link['ata'] = train_head['ata']
            data_link['distance'] = train_head['distance']
            data_link['simple_eta'] = train_head['simple_eta']
            print("flag:", 3)
            data_link['slice_id'] = train_head['slice_id']
            print("flag:", 4)
            data_link_split = data_link[1].str.split(' ', expand=True).stack().to_frame()
            print("flag:", 5)
            data_link_split = data_link_split.reset_index(level=1, drop=True).rename(columns={0: 'link_info'})
            print("flag:", 6)
            data_link_split = data_link[['order_id', 'index', 'ata', 'distance', 'simple_eta', 'slice_id']].join(
                data_link_split)
            print("flag:", 7)
            data_link_split = data_link_split.reset_index(drop=True)
            data_link_split[['link_id',
                             'link_time',
                             'link_ratio',
                             'link_current_status',
                             'link_arrival_status']] = data_link_split['link_info'].str.split(':|,', 5, expand=True)
            print("flag:", 8)
            data_link_split = data_link_split[['order_id','link_id']]
            data_link_split['link_id'] = data_link_split['link_id'].astype(int)
            features = pd.DataFrame({'order_id': data_link_split['order_id'].unique()})
            groupby = data_link_split.groupby(['order_id'])
            func = partial(link_id_find)
            g = parallel_apply(groupby, func, index_name='order_id', num_workers=5, chunk_size=10000)
            g = pd.DataFrame(g,columns=['from_id','to_id'])
            g = g.drop_duplicates()
            nextlinks_new.append(g)
        nextlinks_new = pd.concat(nextlinks_new, axis=0)
        nextlinks_new = nextlinks_new.drop_duplicates()
        nextlinks_new = nextlinks_new.sort_values(by='from_id').reset_index(drop=True)
        nextlinks = pd.concat([nextlinks,nextlinks_new],axis=0)
        nextlinks = nextlinks.drop_duplicates()
        nextlinks = nextlinks.sort_values(by='from_id').reset_index(drop=True)
        print('save all csv')
        nextlinks.to_csv(root_path+'nextlinks_allday.csv',index=False)
    print('calcute weight')
    nextlinks = nextlinks.sort_values(by='from_id').reset_index(drop=True)
    nextlinks = nextlinks.drop_duplicates()
    from_id_weight = nextlinks['from_id'].value_counts()
    from_id_weight = from_id_weight.to_frame()
    from_id_weight['index'] = from_id_weight.index
    from_id_weight.columns = ['weight', 'from_id']
    nextlinks = pd.merge(nextlinks, from_id_weight, 'left', on=['from_id'])
    print('calcute weight finish!')
    nextlinks['to_id'] = nextlinks['to_id'].astype(int)
    nextlinks['from_id'] = nextlinks['from_id'].astype(int)
    id_key = list(set(nextlinks['from_id'].unique().tolist() + nextlinks['to_id'].unique().tolist()))
    id_key_to_connected = dict(zip(id_key, range(len(id_key))))
    nextlinks['from_id'] = nextlinks['from_id'].map(id_key_to_connected)
    nextlinks['to_id'] = nextlinks['to_id'].map(id_key_to_connected)
    np.save(root_path + 'id_key_to_connected_allday.npy', id_key_to_connected)
    print('id key save finish!')
    print('start creating graph')
    G = nx.DiGraph()
    from_id = nextlinks['from_id'].to_list()
    to_id = nextlinks['to_id'].to_list()
    weight = nextlinks['weight'].to_list()
    edge_tuple = list(zip(from_id, to_id,weight))
    # edge_tuple = tuple(from_id,to_id,weight)
    print('adding')
    G.add_weighted_edges_from(edge_tuple)
    G = G.to_directed()
    print('finish create graph!')
    print('start train n2v')
    look_back = list(G.nodes())
    embeddings = {}
    models = []
    models.append(node2vec(d=128, max_iter=10, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1))
    for embedding in models:
        Y, t = embedding.learn_embedding(graph=G, edge_f=None,
                              is_weighted=True, no_python=True)
        for i, embedding in enumerate(embedding.get_embedding()):
            embeddings[look_back[i]] = embedding
    np.save(root_path+'graph_embeddings_retp1.npy', embeddings)
    print('nextlink graph embedding retp 1 finish!') # displays "world"
    del models
    gc.collect()

    look_back = list(G.nodes())
    embeddings = {}
    models = []
    models.append(node2vec(d=128, max_iter=10, walk_len=80, num_walks=10, con_size=10, ret_p=0.5, inout_p=1))
    for embedding in models:
        Y, t = embedding.learn_embedding(graph=G, edge_f=None,
                                         is_weighted=True, no_python=True)
        for i, embedding in enumerate(embedding.get_embedding()):
            embeddings[look_back[i]] = embedding
    np.save(root_path + 'graph_embeddings_retp05.npy', embeddings)
    print('nextlink graph embedding retp 0.5 finish!')

