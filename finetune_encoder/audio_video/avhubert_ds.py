from torch.utils.data import Dataset, Subset
import os
AUDIORATE=16000
import os
import pandas as pd
import numpy as np
import torch 
import torch.nn.functional as F
from scipy.io import wavfile
from python_speech_features import logfbank
from torch.nn.utils.rnn import pad_sequence
import sys
sys.path.append("E:/university/FYT/repos/multi_modal_ser/finetune_encoder/audio_video/av_hubert/avhubert")
import utils as avhubert_utils

class AVHUBERTDataset(Dataset):
    """multi model ser dataset."""
    
    def stacker(self, feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
        return feats
        
    def __load_label__(self, cutmap_path):
        sheet_df = pd.DataFrame()
        for ses in range(1, 6):
            extractionmapPATH = cutmap_path + \
                str(ses)+'.xlsx'
            xl = pd.ExcelFile(extractionmapPATH)
            sheets = xl.sheet_names
            for sheet in sheets:
                sheet_df = pd.concat([sheet_df, xl.parse(sheet)])
        self.df_ = sheet_df
        
        # remove samples not agreed
        self.df_ = pd.merge(self.df_, self.df_text, on=["smp_id"])
        self.df_["emotion_id"] = self.df_["emotion"].map(self.emo2id)
        self.df_ = self.df_[self.df_["emotion_id"].notna()].reset_index(drop=True)
        self.df_["session"] = self.df_["smp_id"].apply(lambda x: x.split("_")[0])
        
    def __load_text__(self, text_path):
        self.df_text = pd.read_csv(text_path)
        pass
    
    def __load_audio__(self, fn_path):
        self.fn_list = list(self.df_["smp_id"])
        self.raw_list = []
        for fn in self.fn_list:
            self.raw_list.append(wavfile.read(os.path.join(fn_path, fn)+'.wav')[1])
    
    def __load_video__(self, idx):
        frames = avhubert_utils.load_video(os.path.join(self.video_path, idx.split("_")[0][:-1], idx+".mp4"))
#         transform = avhubert_utils.Compose([
#           avhubert_utils.Normalize(0.0, 255.0),
#           avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
#           avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)])
        # frames = transform(frames)
        frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0)
        video_feats = frames
        return video_feats
    
    def __init__(self, 
                 fn_path, 
                 cutmap_path, 
                 text_path, 
                 video_path, 
                 emo2id,
                 audio_in_features = 104):
        
        self.emo2id = emo2id
        self.audio_in_features = audio_in_features
        self.video_path = video_path
        self.__load_text__(text_path)
        self.__load_label__(cutmap_path)
        self.__load_audio__(fn_path)
        
    def __len__(self):
        return self.df_.shape[0]
    
    def __getsingle__(self, idx):
        raw_audio = self.raw_list[idx]
        video_feats = self.__load_video__(self.fn_list[idx])
        audio_feats = logfbank(raw_audio, samplerate=AUDIORATE).astype(np.float32) # [T, F]
        audio_feats = self.stacker(audio_feats, self.audio_in_features//26) # [T/stack_order_audio, F*stack_order_audio]

        diff = audio_feats.shape[0] - video_feats.shape[2]
        if diff < 0:
            audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
        elif diff > 0:
            audio_feats = audio_feats[:-diff]

        with torch.no_grad():
            audio_feats = torch.from_numpy(audio_feats.astype(np.float32))
            audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
            audio_feats = audio_feats.unsqueeze(dim=0)
        return audio_feats, video_feats
    
    def collate(self, audio, video, max_size=500):
        padded_audio = pad_sequence([a.squeeze() for a in audio]+[torch.empty(500, 104)], batch_first=True)[:-1]
#         print(padded_audio)
        padded_video = pad_sequence([v.squeeze() for v in video]+[torch.empty(500,88,88)], batch_first=True)[:-1, np.newaxis, : ,:,:]
        mask = torch.zeros_like(padded_audio)
        mask[padded_audio != 0] = 1
        return padded_audio, padded_video, mask
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            ret_dict = {}
            for key in self.__getitem__(0).keys():
                ret_dict[key] = [self.__getitem__(i)[key] for i in range(*idx.indices(len(self)))]
            
#             padded_audio, padded_video, padding_mask = self.collate(ret_dict["audio"], ret_dict["video"])
#             ret_dict["padding_mask"] = padding_mask
#             ret_dict["audio"] = padded_audio.transpose(1, 2)
#             ret_dict["video"] = padded_video

            ret_dict["audio"] = torch.stack(ret_dict["audio"])
            ret_dict["video"] = torch.stack(ret_dict["video"])
            ret_dict["padding_mask"] = torch.stack(ret_dict["padding_mask"])
            return ret_dict
        
        else:
            audio_feats, video_feats = self.__getsingle__(idx)
            padded_audio, padded_video, padding_mask = self.collate([audio_feats], [video_feats])
            
            return {
                "padding_mask": padding_mask[0][:500, :],
                "sess": list(self.df_["session"])[idx],
                "fn": self.fn_list[idx],
                "audio": padded_audio[0][:500, :].T,
                "video": padded_video[0][:, :500, :, :],
                "text": list(self.df_["transcript"])[idx],
                "labels": list(self.df_["emotion_id"])[idx]
            }