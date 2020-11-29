# coding: utf-8

import pandas as pd
import numpy as np
from random import shuffle

def f(table,name='prob'):
    table=table.copy()
    score=[]
    for i in [0.40,0.41,0.42,0.43,0.44,0.45]:
        table['pred']=1*(table[name]>i)
        c=((table.pred==1)&(table.label==1)).sum()
        p=c/table.pred.sum()
        r=c/table.label.sum()
        score.append(2*p*r/(p+r))
    return score

def record_to_sequence(table):
    table.columns=['user_id','day','value']
    table.sort_values(by=['user_id','day'],inplace=True)
    table['string']=table.day.map(str)+':'+table.value.map(str)
    table=table.groupby(['user_id'],as_index=False).agg({'string':lambda x:','.join(x)})
    return table

class user_seq:
    
    def __init__(self,register_day,seq_length,n_features):
        self.register_day=register_day
        self.seq_length=seq_length
        self.array=np.zeros([self.seq_length,n_features])
        self.array[0,0]=1
        self.page_rank=np.zeros([self.seq_length])
        self.pointer=1
        
    def put_feature(self,feature_number,string):
        for i in string.split(','):
            pos,value=i.split(':')
            self.array[int(pos)-self.register_day,feature_number]=1

    def put_PR(self,string):
        for i in string.split(','):
            pos,value=i.split(':')
            self.page_rank[int(pos)-self.register_day]=value

    def get_array(self):
        return self.array
    
    def get_label(self):
        self.label=np.array([None]*self.seq_length)
        active=self.array[:,:10].sum(axis=1)
        for i in range(self.seq_length-7):
            self.label[i]=1*(np.sum(active[i+1:i+8])>0)
        return self.label
    

class DataGenerator:
    
    def __init__(self,register,launch,create,activity):
        
        register=register.copy()
        launch=launch.copy()
        create=create.copy()
        activity=activity.copy()
        
        #user_queue
        register['seq_length']=31-register['register_day']
        self.user_queue={i:[] for i in range(1,31)}
        for index,row in register.iterrows():
            self.user_queue[row[-1]].append(row[0]) #row[-1]是seq_length,row[0]是user_id
        
        #初始化self.data
        n_features=12 #row[0]是user_id,row[1]是register_day,row[-1]是seq_length
        self.data={row[0]:user_seq(register_day=row[1],seq_length=row[-1],n_features=n_features) for index,row in register.iterrows()}
        

        #提取launch_seq
        launch['launch']=1
        launch_table=launch.groupby(['user_id','launch_day'],as_index=False).agg({'launch':'sum'})
        launch_table=record_to_sequence(launch_table)
        for index,row in launch_table.iterrows():
            self.data[row[0]].put_feature(1,row[1]) #row[0]是user_id,row[1]是string
            
        #提取create_seq
        create['create']=1
        create_table=create.groupby(['user_id','create_day'],as_index=False).agg({'create':'sum'})
        create_table=record_to_sequence(create_table)
        for index,row in create_table.iterrows():
            self.data[row[0]].put_feature(2,row[1]) #row[0]是user_id,row[1]是string

        #提取act_seq
        for i in range(6):
            act=activity[activity.act_type==i].copy()
            act=act.groupby(['user_id','act_day'],as_index=False).agg({'video_id':'count'})
            act=record_to_sequence(act)
            for index,row in act.iterrows():
                self.data[row[0]].put_feature(i+3,row[1]) #row[0]是user_id,row[1]是string

        #提取page_seq
        for i in range(1):
            act=activity[activity.page==i].copy()
            act=act.groupby(['user_id','act_day'],as_index=False).agg({'video_id':'count'})
            act=record_to_sequence(act)
            for index,row in act.iterrows():
                self.data[row[0]].put_feature(i+9,row[1]) #row[0]是user_id,row[1]是string

        #提取watched
        watched=register.loc[:,['user_id']].copy()
        watched.columns=['author_id']
        watched=pd.merge(watched,activity[activity.author_id!=activity.user_id],how='inner')
        watched=watched.groupby(['author_id','act_day'],as_index=False).agg({'video_id':'count'})
        watched=record_to_sequence(watched)
        for index,row in watched.iterrows():
            self.data[row[0]].put_feature(10,row[1]) #row[0]是user_id,row[1]是string

        #提取watched by self
        watched=activity[activity.author_id==activity.user_id].copy()
        watched=watched.groupby(['user_id','act_day'],as_index=False).agg({'video_id':'count'})
        watched=record_to_sequence(watched)
        for index,row in watched.iterrows():
            self.data[row[0]].put_feature(11,row[1]) #row[0]是user_id,row[1]是string

        #提取label
        self.label={user_id:user.get_label() for user_id,user in self.data.items()}
        
        #提取data
        self.data={user_id:user.get_array() for user_id,user in self.data.items()}


        #set sample strategy
        self.local_random_list=[]
        for i in range(15,31):
            self.local_random_list+=[i]*(i-14)
            
        self.online_random_list=[]
        for i in range(8,31):
            self.online_random_list+=[i]*(i-7)

        self.local_train_list=list(range(15,31))
        self.local_test_list=list(range(8,31))
        self.online_train_list=list(range(8,31))
        self.online_test_list=list(range(1,31))

        self.pointer={i:0 for i in range(1,31)}
        
    
    def reset_pointer(self):
        self.pointer={i:0 for i in range(1,31)}
        
        
    def next_batch(self,batch_size=1000):

        seq_length=self.local_random_list[np.random.randint(len(self.local_random_list))]
        batch_size=batch_size//(seq_length-14)+1

        if self.pointer[seq_length]+batch_size>len(self.user_queue[seq_length]):
            self.pointer[seq_length]=0
            shuffle(self.user_queue[seq_length])
            #print('---------------------',seq_length,'shuffled ------------------------------')
        start=self.pointer[seq_length]
        user_list=self.user_queue[seq_length][start:start+batch_size]
        self.pointer[seq_length]+=batch_size

        user_matrix=np.array(user_list)
        data_matrix=np.array([self.data[i] for i in user_list])
        label_matrix=np.array([self.label[i] for i in user_list])
        
        return seq_length,user_matrix,data_matrix,label_matrix
    
    
    def get_set(self,usage='train'):
        
        if usage=='train':
            test_list=self.local_train_list
        else:
            test_list=self.local_test_list
        
        user_list=[np.array(self.user_queue[seq_length]) for seq_length in test_list]
        data_list=[np.array([self.data[user_id] for user_id in self.user_queue[seq_length]]) for seq_length in test_list]
        label_list=[np.array([self.label[user_id] for user_id in self.user_queue[seq_length]]) for seq_length in test_list]
        return test_list,user_list,data_list,label_list