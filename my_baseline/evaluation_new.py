#coding=utf-8
from __future__ import print_function
import os
import pickle

import numpy
import time
import numpy as np
import torch
import pdb
import tqdm
import sys

sys.path.insert(0,'../')
from my_baseline.model import VSE
from collections import OrderedDict
from my_baseline.dataloader import get_val_dataloader,get_test_dataloader
from my_baseline.options import options
from vocab.vocab import Vocabulary


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


def encode_data(model, data_loader):

    model.val_start()

    # numpy array to keep all the embeddings
    vid_embs_1 = None
    # vid_embs_2 = None
    # cap_embs = None

    index = 0
    batch_size = data_loader.batch_size
    for i, (video_feat1,video_feat2, captions, lengths, ids) in enumerate(data_loader):

        video_feat_1, cap_emb_1 = model.forward_emb(video_feat1, captions, lengths)
        if vid_embs_1 is None:
            vid_embs_1 = np.zeros((len(data_loader.dataset), video_feat_1.size(1)))
            cap_embs_1 = np.zeros((len(data_loader.dataset), cap_emb_1.size(1)))

        # ids 需要更改
        vid_embs_1[index:index+batch_size] = video_feat_1.data.cpu().numpy().copy()
        cap_embs_1[index:index+batch_size] = cap_emb_1.data.cpu().numpy().copy()

        index += batch_size

    return vid_embs_1, cap_embs_1


def encode_data_2(model, data_loader):
    model.val_start()
    index = 0
    batch_size = data_loader.batch_size

    vid_embs_1 = None

    for i, (video_feat1, video_feat2, captions, lengths, ids) in enumerate(data_loader):
       video_feat_1, cap_emb_1 = model.forward_emb(video_feat2, captions, lengths)
       if vid_embs_1 is None:
            vid_embs_1 = np.zeros((len(data_loader.dataset), video_feat_1.size(1)))
            cap_embs_1 = np.zeros((len(data_loader.dataset), cap_emb_1.size(1)))
       vid_embs_1[index:index + batch_size] = video_feat_1.data.cpu().numpy().copy()
       cap_embs_1[index:index + batch_size] = cap_emb_1.data.cpu().numpy().copy()
       index += batch_size

    return vid_embs_1, cap_embs_1

def i2t(vid_feats_1, captions_1, return_ranks=False):
    # pdb.set_trace()
    npts = vid_feats_1.shape[0] / 20
    index_list = []
    print(npts)

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):
        vid_feat_1 = vid_feats_1[20 * index].reshape(1, vid_feats_1.shape[1])
        # vid_feat_2 = vid_feats_2[20 * index].reshape(1, vid_feats_2.shape[1])

        d1 = numpy.dot(vid_feat_1, captions_1.T).flatten()
        # d2 = numpy.dot(vid_feat_2, captions_2.T).flatten()
        # d = d1 + d2
        d = d1

        inds = numpy.argsort(d)[::-1] # from large to samll
        index_list.append(inds[0])

        rank = 1e20

        for i in range(20 * index, 20 * index + 20, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
                flag=i-20 * index
        ranks[index] = rank           # 寻找ground_truth中最近的一个句子
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(vid_feats_1, captions_1, return_ranks=False):
    """
    Text->Videos (Video Search)
    Videos: (20N, K) matrix of videos
    Captions: (20N, K) matrix of captions
    """

    npts = vid_feats_1.shape[0] / 20
    vid_feat_1 = numpy.array([vid_feats_1[i] for i in range(0, len(vid_feats_1), 20)])
    # vid_feat_2 = numpy.array([vid_feats_2[i] for i in range(0, len(vid_feats_2), 20)])

    ranks = numpy.zeros(20 * npts)
    top1 = numpy.zeros(20 * npts)
    for index in range(npts):
        # Get query captions
        queries_1 = captions_1[20 * index:20 * index + 20]
        # queries_2 = captions_2[20 * index:20 * index + 20]


        d1 = numpy.dot(queries_1, vid_feat_1.T)
        # d2 = numpy.dot(queries_2, vid_feat_2.T)
        # d = d1 + d2
        d = d1

        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[20 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[20 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def encode_eval(model_1, model_2, data_loader):
    model_1.val_start()
    model_2.val_start()
    index = 0
    batch_size = data_loader.batch_size

    vid_embs_1 = None

    for i, (video_feat1, video_feat2, captions, lengths, ids) in enumerate(data_loader):
        video_feat_1, cap_emb_1 = model_1.forward_emb(video_feat1, captions, lengths)
        video_feat_2, cap_emb_2 = model_2.forward_emb(video_feat2, captions, lengths )
        if vid_embs_1 is None:
            vid_embs_1 = np.zeros((len(data_loader.dataset), video_feat_1.size(1)))
            cap_embs_1 = np.zeros((len(data_loader.dataset), cap_emb_1.size(1)))
            vid_embs_2 = np.zeros((len(data_loader.dataset), video_feat_2.size(1)))
            cap_embs_2 = np.zeros((len(data_loader.dataset), cap_emb_2.size(1)))

        vid_embs_1[index:index + batch_size] = video_feat_1.data.cpu().numpy().copy()
        vid_embs_2[index:index + batch_size] = video_feat_2.data.cpu().numpy().copy()
        cap_embs_1[index:index + batch_size] = cap_emb_1.data.cpu().numpy().copy()
        cap_embs_2[index:index + batch_size] = cap_emb_2.data.cpu().numpy().copy()
        index += batch_size

    return vid_embs_1, vid_embs_2, cap_embs_1, cap_embs_2


def i2t_eval(vid_feats_1,vid_feats_2,captions_1,captions_2, return_ranks=False):
    # pdb.set_trace()
    npts = vid_feats_1.shape[0] / 20
    index_list = []
    print(npts)

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):
        vid_feat_1 = vid_feats_1[20 * index].reshape(1, vid_feats_1.shape[1])
        vid_feat_2 = vid_feats_2[20 * index].reshape(1, vid_feats_2.shape[1])

        d1 = numpy.dot(vid_feat_1, captions_1.T).flatten()
        d2 = numpy.dot(vid_feat_2, captions_2.T).flatten()
        d = d1 + d2
        # d = d1

        inds = numpy.argsort(d)[::-1] # from large to samll
        index_list.append(inds[0])

        rank = 1e20

        for i in range(20 * index, 20 * index + 20, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
                flag=i-20 * index
        ranks[index] = rank           # 寻找ground_truth中最近的一个句子
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def t2i_eval(vid_feats_1,vid_feats_2, captions_1,captions_2, return_ranks=False):
    """
    Text->Videos (Video Search)
    Videos: (20N, K) matrix of videos
    Captions: (20N, K) matrix of captions
    """
    npts = vid_feats_1.shape[0] / 20
    vid_feat_1 = numpy.array([vid_feats_1[i] for i in range(0, len(vid_feats_1), 20)])
    vid_feat_2 = numpy.array([vid_feats_2[i] for i in range(0, len(vid_feats_2), 20)])

    ranks = numpy.zeros(20 * npts)
    top1 = numpy.zeros(20 * npts)
    for index in range(npts):
        # Get query captions
        queries_1 = captions_1[20 * index:20 * index + 20]
        queries_2 = captions_2[20 * index:20 * index + 20]
        d1 = numpy.dot(queries_1, vid_feat_1.T)
        d2 = numpy.dot(queries_2, vid_feat_2.T)
        d = d1 + d2

        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[20 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[20 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def evalrank(model_1_path, model_2_path, data_path = None, split = 'dev', opt=options()):

    model = torch.load(model_1_path)
    model_2 = torch.load(model_2_path)
    # opt = check_point['opt']
    # print(opt)

    if data_path is not None:
        opt.data_path = data_path
    opt['vocab_path'] = '../vocab'

    vocab = pickle.load(open(os.path.join(opt['vocab_path'],'vocab.pkl'),'rb'))

    opt['vocab_size'] = len(vocab)

    # model = VSE(opt)

    # model.load_state_dict(check_point['model'])

    print('Loading dataset')

    data_loader = get_test_dataloader(opt=opt)

    # vid_feats_1, cap_embs_1 = encode_data(model, data_loader)
    # vid_feats_2, cap_embs_2 = encode_data_2(model_2, data_loader)
    vid_feats_1,vid_feats_2,cap_embs_1,cap_embs_2 = encode_eval(model,model_2,data_loader)

    print('Videos: %d, Captions: %d' %
          (vid_feats_1.shape[0] / 20, cap_embs_1.shape[0]))

    r, rt = i2t_eval(vid_feats_1,vid_feats_2,cap_embs_1,cap_embs_2,return_ranks=True)
    ri, rti = t2i_eval(vid_feats_1,vid_feats_2,cap_embs_1,cap_embs_2,return_ranks=True)

    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: " , rsum)
    print("Average i2t Recall: ", ar)
    print("Video to text: ", r)
    print("Average t2i Recall: ", ari)
    print("Text to Video: ", ri)

    # torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')
    return r,ri

if __name__ == '__main__':
    train_id = '3'
    epoch = 10
    model_path = './model/train_' + str(train_id) + '_epoch_'+str(epoch)+'.pth'
    model = torch.load(model_path)
    r,ri = evalrank(model)
    pdb.set_trace()
    print('ss')
