import os
os.environ["TORCH_HOME"]="/data/chuak"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,5,6"
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
import cv2
import tempfile
import torch
from transformers import Trainer, TrainingArguments
import sys
sys.path.append("/data/chuak/mmser/src/multi_modal_ser/finetune_encoder/audio_video/av_hubert/avhubert")
# %cd /home/multi_modal_ser/finetune_encoder/audio_video/av_hubert/
import utils as avhubert_utils
from argparse import Namespace
from IPython.display import HTML
import numpy as np
import sys
print(sys.version)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, Subset
import os
import datetime
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
import random
import torch.nn as nn
from avhubert_trainer import CustomTrainer , compute_metrics
from transformers import EarlyStoppingCallback, TrainerCallback, TrainerState
from facenet_pytorch import MTCNN, InceptionResnetV1
from avhubert_classifier import AVHUBERTClassifier

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
                 labels_list, aug_prob=0.3):

        self.label_smp_dict = {}
        for idx, label in enumerate(labels_list):
            if label not in self.label_smp_dict:
                self.label_smp_dict[label] = []
            self.label_smp_dict[label].append({
                "audio": audio_feats_list[idx],
                "video": video_feats_list[idx],
                "text": text_list[idx],
                "label": labels_list[idx],
            })
        self.aug = False
        self.aug_len = 0
        self.sometimes = lambda aug: va.Sometimes(aug_prob, aug) # Used to apply augmentor with 50% probability
        
        self.transform_list = [
            self.sometimes(va.Salt()),
            self.sometimes(va.Pepper()),
            self.sometimes(va.RandomShear(0.2, 0.2)),
            self.sometimes(va.HorizontalFlip()),
            self.sometimes(va.VerticalFlip()),
            self.sometimes(va.RandomRotate(30)),
            self.sometimes(va.GaussianBlur(0.8)),
            self.sometimes(va.ElasticTransformation(0.2,0.2)),
            self.sometimes(va.PiecewiseAffineTransform(20,10,0.5)),
        ]
        

    def __upsample__(self, origin_smp_list, smp_length, k=None):
        smpled_list = origin_smp_list
        for idx in tqdm(range(smp_length-len(origin_smp_list))):
            seq = va.Sequential(random.choices(self.transform_list, k=k))
            smp = random.choice(origin_smp_list)
            vid = smp["video"].squeeze()
            vid = [cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB).astype(np.uint8) for frame in vid]
            vid = np.stack(vid)
            video_aug = seq(vid)
            video_aug = [cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.uint8) for frame in video_aug]
            video_aug = np.stack(video_aug)
            video_aug = video_aug[np.newaxis, np.newaxis]
            
            smpled_list.append({
                "video": video_aug,
                "audio": smp['audio'],
                "text": smp['text'],
                "label": smp['label'],
            })
        return smpled_list
            
        
    
    def __aug__(self, niters=2, nchoice=2, isaug=True):
        self.aug_label_smp_dict = {}
        self.aug_smps = []
        if isaug:
            print([len(v) for k,v in self.label_smp_dict.items()])
            smp_size = max([len(v) for k,v in self.label_smp_dict.items()])*(niters+1)
            for k, v in self.label_smp_dict.items():
                print(smp_size, len(v))
                upsmped = self.__upsample__(v, smp_size, nchoice)
                self.aug_label_smp_dict[k] = upsmped
                self.aug_smps += upsmped
                
            self.aug = True
            print([len(v) for k,v in self.aug_label_smp_dict.items()])
        else:
            print([len(v) for k,v in self.label_smp_dict.items()])
            for k, v in self.label_smp_dict.items():
                self.aug_label_smp_dict[k] = v
                self.aug_smps += v
            self.aug = True
            print([len(v) for k,v in self.aug_label_smp_dict.items()])
            
    
    def __len__(self):
        return len(self.aug_smps)
    
    def __getitem__(self, idx):
        audio_feats = self.aug_smps[idx]["audio"]
        video_feats = self.aug_smps[idx]["video"]/255
        padded_audio, padded_video, padding_mask = self.collate([audio_feats], [torch.tensor(video_feats)])
        return {
            "padding_mask": padding_mask[0][:500, :].float(),
            "audio": padded_audio[0][:500, :].T.float(),
            "video": padded_video[0][:, :500, :, :].float(),
            "text": self.aug_smps[idx]["text"],
            "labels": self.aug_smps[idx]["label"]
        }


class FaceNetBlock(nn.Module):
    
    def __init__(self, in_channels, facenet):
        super(FaceNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 3
        self.facenet = facenet
    def forward(self, x):
        x = self.facenet(x)
        return x

class FreezingCallback(TrainerCallback):

    def __init__(self, freeze_encoder_epochs: int):
        self.freeze_encoder_epochs = freeze_encoder_epochs

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(state.epoch, self.freeze_encoder_epochs)
        model = kwargs["model"]
        if state.epoch >= self.freeze_encoder_epochs:
            print("="*10, "Freezing", "="*10)
            for param in model.encoder.feature_extractor_video.parameters():
                param.requires_grad = False

    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        for name, param in model.named_parameters():
            param.requires_grad = True

def main():
    from avhubert_ds import AVHUBERTDataset
    mmser_ds = torch.load("/data/chuak/mmser/data/avhubert_ds2.pt")
    mmser_ds.video_path = "/data/chuak/mmser/data/roi/"
    mmser_ds.cached = False
    mmser_ds.__cache__()
    meta_df_ = mmser_ds.df_
    mmser_ds.df_["bigsess"] = mmser_ds.df_["session"].apply(lambda x: x[:-1])
    sess_dict = mmser_ds.df_.groupby("bigsess").groups
    all_indices = set(mmser_ds.df_.index.tolist())

    audio_feats_list = mmser_ds.audio_feats_list
    video_feats_list = mmser_ds.video_feats_list
    text_list = list(meta_df_["transcript"])
    labels_list = list(meta_df_["emotion_id"])
    
    del mmser_ds
    val_indices = sess_dict['Ses03']
    test_indices = sess_dict['Ses04']
    train_indices = list(all_indices-set(val_indices)-set(test_indices))
    train_ds = VidaugDataset(
        [item.detach().numpy() for i, item in enumerate(audio_feats_list) if i in train_indices],
        [item.detach().numpy() for i, item in enumerate(video_feats_list) if i in train_indices],
        [item for i, item in enumerate(text_list) if i in train_indices],
        [item for i, item in enumerate(labels_list) if i in train_indices]
    )
    
    val_ds = VidaugDataset(
        [item.detach().numpy() for i, item in enumerate(audio_feats_list) if i in val_indices],
        [item.detach().numpy() for i, item in enumerate(video_feats_list) if i in val_indices],
        [item for i, item in enumerate(text_list) if i in val_indices],
        [item for i, item in enumerate(labels_list) if i in val_indices]
    )
    
    test_ds = VidaugDataset(
        [item.detach().numpy() for i, item in enumerate(audio_feats_list) if i in test_indices],
        [item.detach().numpy() for i, item in enumerate(video_feats_list) if i in test_indices],
        [item for i, item in enumerate(text_list) if i in test_indices],
        [item for i, item in enumerate(labels_list) if i in test_indices]
    )
    train_ds.__aug__(2)
    val_ds.__aug__(isaug=False)
    test_ds.__aug__(isaug=False)

    user_dir = "/data/chuak/mmser/src/multi_modal_ser/finetune_encoder/audio_video/av_hubert/avhubert"
    utils.import_user_module(Namespace(user_dir=user_dir))
    ckpt_path = "/data/chuak/mmser/check_pts/avhubert.pt"
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])  
    model = models[0]
    if hasattr(models[0], 'decoder'):
        print(f"Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model
    else:
        print(f"Checkpoint: pre-trained w/o fine-tuning")

    import json
    # avhubert_classifier = AVHUBERTClassifier(model, 768, 256)
    avhubert_classifier = AVHUBERTClassifier(model, 768, 64)
    for param in avhubert_classifier.parameters():
        param.requires_grad = True
    
    wandb.init()
    train_set = train_ds
    val_set = val_ds
    test_set = test_ds

    # avhubert_classifier.encoder.feature_extractor_video.resnet = InceptionResnetV1(pretrained='vggface2')
    avhubert_classifier = avhubert_classifier.to(device)
    avhubert_classifier = AVHUBERTClassifier(model, 768, 32)
    
    facenet_res = InceptionResnetV1(pretrained='vggface2')

    avhubert_classifier.encoder.feature_extractor_video.resnet.trunk = facenet_res
    avhubert_classifier.encoder.feature_extractor_video.resnet.frontend3D =nn.Conv3d(1, 3, 
                                          kernel_size=(5, 7, 7), 
                                          stride=(1, 1, 1), 
                                          padding=(2, 3, 3), bias=False)
    for param in avhubert_classifier.encoder.feature_extractor_video.resnet.trunk.parameters():
        param.requires_grad = False

    output_dir=os.path.join("check_pts", "AVHUBERT", datetime.datetime.now().date().strftime(format="%Y-%m-%d"))
    
    training_args = TrainingArguments(output_dir,report_to="wandb")
    training_args.remove_unused_columns=False
    training_args.per_device_train_batch_size=2
    training_args.per_device_eval_batch_size=2
    training_args.logging_steps = 200 # int(1000/training_args.per_device_train_batch_size/8)
    training_args.eval_steps = 200 # int(1000/training_args.per_device_train_batch_size/8)
    training_args.evaluation_strategy="steps" 
    training_args.logging_strategy="steps"
    training_args.load_best_model_at_end=True,
    training_args.save_strategy = "no"
    training_args.learning_rate=5e-4
    training_args.num_train_epochs=7
    training_args.metric_for_best_model = 'accuracy'

    
    trainer = CustomTrainer(
        model=avhubert_classifier,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model("/data/chuak/mmser/models/20231217.pt")

   
    eval_result = trainer.evaluate()
    test_result = trainer.predict(test_set).metrics
    
    with open('/data/chuak/mmser/result.txt', 'w') as f:
        f.write(str(eval_result))
        f.write(str(test_result))

if __name__ == "__main__":
    main()
