#coding=utf-8
import cPickle
import pdb
import os
from vocab.vocab import Vocabulary

path = '../vocab/vocab.pkl'
print os.path.exists(path)
vocab_data = cPickle.load(open(path,'r'))
idx2word = vocab_data.idx2word
word2idx = vocab_data.word2idx


# vocab： Vocabulary class
pdb.set_trace()
print 'ss'

