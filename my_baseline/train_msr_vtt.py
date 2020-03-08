#codingutf-8
import pickle
import os
import time
import shutil
import sys

import torch
import logging
import numpy as np
import random
import pdb
from visdom import Visdom

sys.path.insert(0,'../')
from my_baseline.model import VSE
from my_baseline.options import options
from my_baseline.dataloader import get_train_dataloader,get_val_dataloader,get_test_dataloader
from my_baseline.evaluation_new import encode_data,i2t,t2i,evalrank,encode_data_2

from vocab.vocab import Vocabulary

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(opt):
    setup_seed(100)

    train_id = opt['train_id']
    vi = Visdom(env='Video_retrive_'+str(train_id))
    # Load Vocabulary Wrapper
    vocab = pickle.load(open(os.path.join(
        opt['vocab_path'], 'vocab.pkl'), 'rb'))
    opt['vocab_size'] = len(vocab)
    opt['word_nums'] = opt['vocab_size']

    train_loader = get_train_dataloader(opt)
    val_loader = get_val_dataloader(opt)

    model = VSE(opt)

    model_2 = VSE(opt)

    best_score = 0
    best_epoch = 0
    best_score_2 = 0
    best_epoch_2 = 0

    for epoch in range(opt['num_epoch']):
        temp_lr = adjust_learning_rate(opt, model.optimizer, epoch)

        print('train_epoch: ',epoch)
        print('learning rate: ', temp_lr)
        train_epoch(opt, train_loader, model)
        epoch_loss = sum(model.batch_loss)/len(model.batch_loss)
        model.batch_loss = []
        model.epoch_loss.append(epoch_loss)
        print('finish the training epoch of model_1:',epoch)
        print('Current trip loss of model_1 : ', model.epoch_loss[-1])

        currscore = validate(opt,model,val_loader)
        if currscore > best_score:
            best_score = currscore
            best_epoch = epoch
            torch.save(model,'./model_1/train_'+opt['train_id']+'_epoch_'+str(epoch)+'.pth')

        train_epoch(opt, train_loader,model_2,flag=2)
        epoch_loss = sum(model_2.batch_loss) / len(model_2.batch_loss)
        model_2.batch_loss = []
        model_2.epoch_loss.append(epoch_loss)
        print('finished thee training peoch of model_2:', epoch)
        print('Current trip loss of model_2 : 0', model_2.epoch_loss[-1])

        currscore_2 = validate_2(opt, model_2, val_loader)
        if currscore_2 > best_score_2:
            best_score_2 = currscore_2
            best_epoch_2 = epoch
            torch.save(model_2, './model_2/train_' + opt['train_id'] + '_epoch_' + str(epoch) + '.pth')

        path_1 = './model_1/train_'+opt['train_id']+'_epoch_'+str(best_epoch)+'.pth'
        path_2 = './model_2/train_' + opt['train_id'] + '_epoch_' + str(best_epoch_2) + '.pth'
        r,ri = evalrank(path_1,path_2)
        vi.line(X=np.array([epoch]),Y=np.array([r[:3]]),win='video2text',update='append' if epoch>0 else None,
                opts={'title':'video2text','legend':['r1','r5','r10']})
        vi.line(X=np.array([epoch]), Y=np.array([ri[:3]]), win='text2video', update='append' if epoch > 0 else None,
                opts={'title': 'text2video', 'legend': ['r1', 'r5', 'r10']})


def train_epoch(opt, train_loader, model, flag = 0):
    model.train_start()
    if flag==0:
        for i, train_data in enumerate(train_loader):
            model.train_embed(*train_data)
    else:
        for i, train_data in enumerate(train_loader):
            model.train_embed_2(*train_data)


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt['learning_rate'] * (0.5 ** (epoch // opt['lr_update']))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def validate(opt, model, val_loader):

    model.val_start()
    data_loader = val_loader

    vid_feats_1, cap_embs_1 = encode_data(model, data_loader)

    print('Videos: %d, Captions: %d' %
          (vid_feats_1.shape[0] / 20, cap_embs_1.shape[0]))

    (r1, r5, r10, medr, meanr)  = i2t(vid_feats_1, cap_embs_1)
    (r1i, r5i, r10i, medri, meanr) = t2i(vid_feats_1, cap_embs_1)

    print('i2t:')
    print('r1, r5, r10, medr, meanr: ',r1, r5, r10, medr, meanr)
    print('t2i:')
    print('r1i, r5i, r10i, medri, meanr: ',r1i, r5i, r10i, medri, meanr)
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    return currscore

def validate_2(opt, model, val_loader):

    model.val_start()
    data_loader = val_loader

    vid_feats_1, cap_embs_1 = encode_data_2(model, data_loader)

    print('Videos: %d, Captions: %d' %
          (vid_feats_1.shape[0] / 20, cap_embs_1.shape[0]))

    (r1, r5, r10, medr, meanr)  = i2t(vid_feats_1, cap_embs_1)
    (r1i, r5i, r10i, medri, meanri) = t2i(vid_feats_1, cap_embs_1)

    print('i2t:')
    print('r1, r5, r10, medr, meanr: ',r1, r5, r10, medr, meanr)
    print('t2i:')
    print('r1i, r5i, r10i, medri, meanr: ',r1i, r5i, r10i, medri, meanri)
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    return currscore


def main():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info('ss')

    opt = options()
    opt['train_id'] = '7'
    opt['train_batch_size'] = 128
    train(opt)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()





if __name__=='__main__':
    main()