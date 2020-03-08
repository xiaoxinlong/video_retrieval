#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.init

from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

def l2norm(X):
    """
    :param X:  bs * dim
    :return:   bs * dim
    """
    norm = torch.pow(X,2).sum(dim=1).sqrt()
    X = X / norm[:,None]

    return X

class Embedding_model(nn.Module):

    def __init__(self,opt):
        super(Embedding_model, self).__init__()
        word_nums = opt['word_nums']
        embedding_dim = opt['embedding_dims']
        if opt['embed_type'] == 'standard':
            self.embed = nn.Embedding(num_embeddings=word_nums,embedding_dim=embedding_dim)
        elif opt['embed_type'] == 'pre_train':   # 使用预训练的词向量，待实现
            self.embed = nn.Embedding(num_embeddings=word_nums, embedding_dim=embedding_dim)

    def forward(self, words):
        return self.embed(words)

class Sentence_embed(nn.Module):
    def __init__(self, opt):
        super(Sentence_embed, self).__init__()
        self.rnn_type = opt['rnn_type']

        self.input_size = opt['input_size']
        self.hidden_szie = opt['hidden_size']

        if self.rnn_type == 'LSTM' or 'lstm':
            self.rnn = nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_szie,num_layers=1,batch_first=True)
        elif self.rnn_type == 'GRU' or 'gru':
            self.rnn = nn.GRU(input_size=self.input_size,hidden_size=self.hidden_szie,batch_first=True)

        self.embed_size = opt['embedding_dims']
        self.embed_model = Embedding_model(opt=opt)

    def forward(self, sentences, length):
        """
        :param sentences: bs*length
        :return: bs*dim
        """
        embed_sentence = self.embed_model(sentences)

        packed = pack_padded_sequence(x,length,batch_first=True)

        out, _ = self.rnn(packed)

        padded = pad_packed_sequence(out,batch_first=True)

        I = torch.LongTensor(length).view(-1,1,1)
        I = I.expand(sentences.size(0),1,self.embed_size-1).cuda()

        out = torch.gather(padded[0],1,I).squeeze(1)

        out = l2norm(out)

        return out


class EncoderVideo(nn.Module):
    def __init__(self, opt):
        self.embed_size = opt['embed_size']
        self.video_dim_1 = opt['video_dim_1']
        self.video_dim_2 = opt['video_dim_2']

        self.fc_1 = nn.Linear(self.video_dim_1,self.embed_size)
        self.fc_2 = nn.Linear(self.video_dim_2,self.embed_size)

        self.video_norm = opt['video_norm']
        self.used_abs = opt['used_abs']

    def forward(self, feature_1, feature_2):

        out_1 = self.fc_1(feature_1)
        out_2 = self.fc_2(feature_2)

        if self.video_norm:
            out_1 = l2norm(out_1)
            out_2 = l2norm(out_2)
        if self.used_abs:
            out_1 = torch.abs(out_1)
            out_2 = torch.abs(out_2)

        return out_1,out_2

class VSE(object):
    def __init__(self,opt):
        self.data = []
        self.video_enc = EncoderVideo(opt)
        self.text_enc = Sentence_embed(opt)

    def train_start(self):
        self.video_enc.train()
        self.text_enc.train()

    def val_start(self):
        self.video_enc.eval()
        self.text_enc.eval()

    def froward_emb(self, video_feat1, video_feat2 , captions, length):
        video_feat1 = video_feat1
        video_feat2 = video_feat2
        captions = captions

        if torch.cuda.is_available():
            video_feat1 = video_feat1.cuda()
            video_feat2 = video_feat2.cuda()
            captions = captions.cuda()

        video_embed_1, video_embed_2 = self.video_enc(video_feat1,video_feat2)
        cap_emb = self.text_enc(captions, length)

        return video_embed_1, video_feat2, cap_emb

def cosine_sim(v,s):
    """
    :param v: bs * dim
    :param s: bs * dim
    :return:  bs * bs
    """
    return v.mm(s.t())

def order_sim(v,s):
    """
    :param v: bs * dim
    :param s: bs * dim
    :return:  bs * bs     max(0,v-s)   clamp 表示将小于min的值全变为min
    """
    temp = (s.unsqueeze(1).expand(s.size(0),v.size(0),s.size(1))
            -v.unsqueeze(1).expand(s.size(0),v.size(0),s.size(1)))

    score = temp.clamp(min=0).pow(2).sum(dim=2).squeeze(2).sqrt().t()

    return score


class Loss(nn.Module):
    def __init__(self,margin=0):
        super(Loss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.beta = 1


    def forward(self, video_feat1, entence_embed):


        scores = self.sim(video_feat1,entence_embed)
        # score_2 = self.sim(video_feat2,entence_embed)
        diagonal = scores.diag().view(video_feat1.size(0),1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        d1_sort, d1_index = torch.sort(scores,dim=1,descending=True)  # from large to small
        val, id1 = torch.min(d1_index,1)
        rank_weights1 = id1.float()   # 这句话并没啥用，只是初始化权重

        for j in range(d1.size(0)):
            rank_weights1[j] = 1 + torch.tensor(self.beta) / (d1.size(0) - (d1_index[j,:]==j).nonzero()).to(dtype=torch.float)

        d2_sort , d2_index = torch.sort(scores.t(), dim=1, descending=True)
        val, id2 = torch.min(d2_index,1)
        rank_weights2 = id2.float()

        for j in range(d2.size(0)):
            rank_weights2[j] = 1 + torch.tensor(self.beta) / (d2.size(0) - (d2_index[j,:]==j).nonzero()).to(dtype=torch.float)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5  # > .5 转换为 torch.uint8类型

        pdb.set_trace()
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        cost_s = torch.mul(rank_weights1, cost_s)
        cost_im = torch.mul(rank_weights2, cost_im)


        return cost_s.sum() + cost_im.sum()



if __name__=='__main__':
    import pdb
    opt = {}
    opt['hidden_size'] = 1024
    opt['rnn_type'] = 'gru'
    opt['input_size'] = 300
    opt['word_nums'] = 10000
    opt['embedding_dims'] = opt['input_size']
    opt['embed_type'] = 'standard'
    opt['vocab'] = None

    # sentence_embed = Sentence_embed(opt=opt)

    loss_fun = Loss(margin=0.5)

    video_feat = l2norm(torch.rand(5,5))
    sentence_feat = l2norm(torch.rand(5,5))

    loss = loss_fun(video_feat,sentence_feat)

    pdb.set_trace()
    print '1'

