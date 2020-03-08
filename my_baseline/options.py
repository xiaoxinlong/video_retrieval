#coding=utf-8
import argparse
from collections import defaultdict

class AttrDict(dict):
    def __init__(self,*args,**kwargs):
        super(AttrDict, self).__init__(*args,**kwargs)
        self.__dict__ = self

def options():
    opt = defaultdict()

    opt['vocab_path'] = '../vocab'
    opt['embed_size'] = 1024    # 最后的向量长度
    opt['embedding_dims'] = 300

    opt['video_dim_1'] = 2048
    opt['video_dim_2'] = 2048
    opt['video_norm'] = True
    opt['used_abs'] = False
    opt['rnn_type'] = 'gru'
    opt['input_size'] = opt['embedding_dims']  # 输入rnn的词向量长度
    opt['hidden_size'] = 512       # rnn 的隐藏层
    opt['embed_type'] = 'standard'
    opt['margin'] = 0.2
    opt['learning_rate'] = 0.0001
    opt['grad_clip'] = 5
    opt['num_epoch'] = 50
    opt['lr_update'] = 10

    opt['video_norm'] = True
    opt['train_id'] = '1'
    opt['sentence_output_size'] = 1024


    return opt