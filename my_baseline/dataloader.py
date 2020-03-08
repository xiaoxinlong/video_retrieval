#coding=utf-8
import numpy
import os
import pdb
import h5py
import cPickle

data_path = '/home/xxl/data/video_retrieval'

# audio_feats = os.path.join(data_path, 'audio_features')
# i3d_feats = os.path.join(data_path,'i3d_features')
# resnet_feats = os.path.join(data_path,'resnet_feature')

def get_audio_feats():
    audio_feats_path = os.path.join(data_path, 'audio_features')
    audio_feats_ids = os.listdir(audio_feats_path)
    audio_feats = {}
    for f in audio_feats_ids[:10]:
        ff = os.path.join(audio_feats_path,f)
        audio_feat_temp = h5py.File(ff,'r')
        audio_feat = audio_feat_temp['layer24'].value
        video_id = f.split('.')[0]
        audio_feats[video_id] = audio_feat
    return audio_feats

def get_i3d_feats():
    i3d_feats_path = os.path.join(data_path, 'i3d_features')
    i3d_feats_ids = os.listdir(i3d_feats_path)
    i3d_feats = {}
    for f in i3d_feats_ids[:10]:
        ff = os.path.join(i3d_feats_path,f)
        i3d_feat = numpy.load(ff)
        video_id = f.split('.')[0].split('-')[-1]
        i3d_feats[video_id] = i3d_feat
    return i3d_feats

def get_resnet_feats():
    resnet_feats_path = os.path.join(data_path, 'resnet_feature')
    resnet_feats_ids = os.listdir(resnet_feats_path)
    resnet_feats = {}
    for f in resnet_feats_ids[:10]:
        ff = os.path.join(resnet_feats_path,f)
        resnet_feat = numpy.load(ff)
        video_id = f.split('.')[0]
        resnet_feats[video_id] = resnet_feat
    return resnet_feats

def get_caption_data():
    caption_pkl_path = os.path.join(data_path,'captions_pkl')
    li = os.listdir(caption_pkl_path)
    # ['msr-vtt_captions_train.pkl', 'msr-vtt_captions_val.pkl', 'msr-vtt_captions_test.pkl']

    train_pkl = os.path.join(caption_pkl_path,'msr-vtt_captions_train.pkl')
    train_data = cPickle.load(open(train_pkl,'r'))

    val_pkl = os.path.join(caption_pkl_path,'msr-vtt_captions_val.pkl')
    val_data = cPickle.load(open(val_pkl,'r'))

    test_pkl = os.path.join(caption_pkl_path,'msr-vtt_captions_test.pkl')
    test_data = cPickle.load(open(test_pkl,'r'))

    # max_length = 0
    # for sentence in train_data[0]:
    #     length = sentence.shape[0]
    #     max_length = max(length,max_length)
    # max_length : 30
    return train_data,val_data,test_data

# train_data[0]   sentences
# train_data[1] sentences_length
# train_data[2] video_id

# i = 100
# assert ((train_data[0][i].shape[0] == train_data[1][i]) or (train_data[0][i].shape[0] == 30))


#  /home/xxl/data/video_retrieval
import torch.utils.data as data
import torch
import numpy as np

# 一个item 包含caption i3d audio
class VTTDataset(data.Dataset):
    def __init__(self,cap_pkl,vid_feature_dir):
        cap_pkl = os.path.join(vid_feature_dir+'/captions_pkl',cap_pkl)
        with open(cap_pkl, 'rb') as f:
            self.captions,self.lengths,self.video_ids = cPickle.load(f)

        self.vid_feat_dir = vid_feature_dir

    def __getitem__(self, index):
        caption = self.captions[index]
        length = self.lengths[index]
        video_id = self.video_ids[index]
        vid_feat_dir = self.vid_feat_dir

        path1 = vid_feat_dir + '/i3d_features/msr_vtt-I3D-RGBFeatures-video'+str(video_id)+'.npy'
        video_feat = torch.from_numpy(np.load(path1))
        video_feat = video_feat.mean(dim=0, keepdim=False) # need some changes. 这里直接用mean操作对于序列数据进行处理

        audio_feat_file = vid_feat_dir + '/audio_features/' + "/video" + str(video_id) + ".mp3.soundnet.h5"
        audio_h5 = h5py.File(audio_feat_file, 'r')
        audio_feat = audio_h5['layer24'][()]
        audio_feat = torch.from_numpy(audio_feat)
        audio_feat = audio_feat.mean(dim=1, keepdim=False)


        video_feat = torch.cat([video_feat, audio_feat])  # 直接将i3d和语音拼接到一起

        path = vid_feat_dir + '/resnet_feature/' + "video" + str(video_id) + ".npy"
        image_feat = torch.from_numpy(np.load(path))
        image_feat = image_feat.mean(dim=0, keepdim=False)  # average pooling
        image_feat = image_feat.float()
        return video_feat.unsqueeze(0), image_feat.unsqueeze(0), caption, index, video_id, length

    def __len__(self):
        return len(self.captions)

def collate_fn(data):
    data.sort(key=lambda x:len(x[2]),reverse=True)
    video_feats,image_feats,captions,indexs,vids,lengths = zip(*data)
    video_feats = torch.cat(video_feats,dim=0)
    image_feats = torch.cat(image_feats,dim=0)
    targets = torch.zeros(len(captions),min(max(lengths),30)).long()
    for i,cap in enumerate(captions):
        end = lengths[i]
        targets[i,:end] = cap
    return video_feats, image_feats, targets, lengths, vids

def collate_fn_train(data):

    data.sort(key=lambda x:len(x[2]),reverse=True)
    video_feats,image_feats,captions,indexs,vids,lengths = zip(*data)
    video_feats = torch.cat(video_feats,dim=0)
    image_feats = torch.cat(image_feats,dim=0)
    targets = torch.zeros(len(captions),min(max(lengths),30)).long()
    for i,cap in enumerate(captions):
        end = lengths[i]
        targets[i,:end] = cap
    video_ids_set = set()
    index = []
    for i,vid in enumerate(vids):
        if vid not in video_ids_set:
            index.append(i)
            video_ids_set.add(vid)
    # return video_feats, image_feats, targets, lengths, vids
    new_lengths = np.array(lengths)[index].tolist()
    new_vids = np.array(vids)[index].tolist()
    return video_feats[index], image_feats[index], targets[index], new_lengths, new_vids

def get_vtt_loader_1(cap_pkl, feature_path, opt, batch_size=2, shuffle=True,train = False):
    v2t = VTTDataset(cap_pkl, feature_path)
    if train:
        collate_funciton = collate_fn_train
    else:
        collate_funciton = collate_fn
    data_loader = torch.utils.data.DataLoader(dataset=v2t,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_funciton)
    return data_loader


def get_train_dataloader(opt):
    train_data_loader = get_vtt_loader_1(cap_pkl='msr-vtt_captions_train.pkl',
                                         feature_path='/home/xxl/data/video_retrieval', opt=opt, batch_size=opt['train_batch_size'], train= True)
    return train_data_loader

def get_val_dataloader(opt):
    val_data_loader = get_vtt_loader_1(cap_pkl='msr-vtt_captions_val.pkl',
                                       feature_path='/home/xxl/data/video_retrieval', opt=opt, batch_size=20,shuffle=False)
    return val_data_loader

def get_test_dataloader(opt):
    test_data_loader = get_vtt_loader_1(cap_pkl='msr-vtt_captions_test.pkl',
                                       feature_path='/home/xxl/data/video_retrieval', opt=opt, batch_size=20, shuffle=False)
    return test_data_loader

if __name__=='__main__':
    # audio = get_audio_feats()
    # i3d = get_i3d_feats()
    resnet = get_resnet_feats()
    # caption_data = get_caption_data()
    opt = {}
    train_data_loader = get_vtt_loader_1(cap_pkl='msr-vtt_captions_train.pkl',feature_path='/home/xxl/data/video_retrieval',opt=opt,batch_size=2)

    val_data_loader = get_vtt_loader_1(cap_pkl='msr-vtt_captions_val.pkl',feature_path='/home/xxl/data/video_retrieval',opt=opt,batch_size=2)

    for i, train_data in enumerate(val_data_loader):
        print i
        pdb.set_trace()

    pdb.set_trace()
    print 'ss'
