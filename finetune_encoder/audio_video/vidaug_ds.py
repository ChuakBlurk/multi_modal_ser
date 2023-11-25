from torch.utils.data import Dataset, Subset
import os
import pandas as pd
import numpy as np
import torch 
import torch.nn.functional as F
import sys
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import vidaug.augmentors as va


class VidaugDataset(Dataset):
    
    def collate(self, audio, video, max_size=500):
        padded_audio = pad_sequence([torch.tensor(a.squeeze()) for a in audio]+[torch.empty(500, 104)], batch_first=True)[:-1]
        padded_video = pad_sequence([v.squeeze().clone().detach() for v in video]+[torch.empty(500,88,88)], batch_first=True)[:-1, np.newaxis, : ,:,:]
        mask = torch.zeros_like(padded_audio)
        mask[padded_audio != 0] = 1
        return padded_audio, padded_video, mask
    
    
    def __init__(self, audio_feats_list, 
                 video_feats_list, 
                 text_list, 
                 labels_list):
        self.audio_feats_list = audio_feats_list
        self.video_feats_list = video_feats_list
        self.text_list = text_list
        self.labels_list = labels_list
    
        # print(len(self.audio_feats_list))
        # print(len(self.video_feats_list))
        # print(len(self.text_list))
        # print(len(self.labels_list))
        
        self.origin_len = len(self.labels_list)
        self.aug_len = 0

    
    
    def __aug__(self, niters=2, aug_prob=0.3, k=None):
        sometimes = lambda aug: va.Sometimes(aug_prob, aug) # Used to apply augmentor with 50% probability
        
        transform_list = [
            # sometimes(va.InvertColor()),
            sometimes(va.Salt()),
            sometimes(va.Pepper()),
            sometimes(va.RandomShear(0.2, 0.2)),
            sometimes(va.HorizontalFlip()),
            sometimes(va.VerticalFlip()),
            sometimes(va.RandomRotate(30)),
            sometimes(va.GaussianBlur(0.8)),
            sometimes(va.ElasticTransformation(0.2,0.2)),
            sometimes(va.PiecewiseAffineTransform(20,10,0.5)),
        ]
        
        
        self.aug_audio_feats_list = []
        self.aug_video_feats_list = []
        self.aug_text_list = []
        self.aug_labels_list = []
        
        for smp_id in tqdm(range(len(self.video_feats_list))):
            for i in range(niters):
                if k is None:
                    seq = va.Sequential(random.choices(transform_list, 
                                                   k=random.choice(range(len(transform_list)))))        
                else:
                    seq = va.Sequential(random.choices(transform_list, 
                                                   k=k))
                vid = self.video_feats_list[smp_id].squeeze()
                # change to color 
                vid = [cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB).astype(np.uint8) for frame in vid]
                vid = np.stack(vid)
                video_aug = seq(vid)
                # print(video_aug[0].shape)
                video_aug = [cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.uint8) for frame in video_aug]
                video_aug = np.stack(video_aug)
                
                # transform = avhubert_utils.Compose([
                #   avhubert_utils.Normalize(0.0, 255.0),
                #   avhubert_utils.CenterCrop((88, 88)),
                #   avhubert_utils.Normalize(0.421, 0.165)])
                # video_aug = transform(video_aug)[np.newaxis, np.newaxis]

                video_aug = (video_aug/255)[np.newaxis, np.newaxis]

                
                video_aug = torch.tensor(video_aug)
                self.aug_audio_feats_list.append(self.audio_feats_list[smp_id])
                self.aug_video_feats_list.append(video_aug)
                self.aug_text_list.append(self.text_list[smp_id])
                self.aug_labels_list.append(self.labels_list[smp_id])
                
        self.aug_len = len(self.aug_labels_list)
                
        
    def __len__(self):
        return self.origin_len + self.aug_len
    
    def __getitem__(self, idx):
        if idx < self.origin_len:
            audio_feats = self.audio_feats_list[idx]
            video_feats = self.video_feats_list[idx]
            padded_audio, padded_video, padding_mask = self.collate([audio_feats], [torch.tensor(video_feats)])
            return {
                "padding_mask": padding_mask[0][:500, :].float(),
                "audio": padded_audio[0][:500, :].T.float(),
                "video": padded_video[0][:, :500, :, :].float(),
                "text": self.text_list[idx],
                "labels": self.labels_list[idx]
            }
        else:
            idx = idx - self.origin_len
            audio_feats = self.aug_audio_feats_list[idx]
            video_feats = self.aug_video_feats_list[idx]
            padded_audio, padded_video, padding_mask = self.collate([audio_feats], [torch.tensor(video_feats)])
            return {
                "padding_mask": padding_mask[0][:500, :].float(),
                "audio": padded_audio[0][:500, :].T.float(),
                "video": padded_video[0][:, :500, :, :].float(),
                "text": self.aug_text_list[idx],
                "labels": self.aug_labels_list[idx]
            }

            