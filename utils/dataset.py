import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MMSERDataset(Dataset):
    """multi model ser dataset."""
    
        
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
        self.df_["emotion_id"] = self.df_["emotion"].map(self.available_emotions)
        self.df_ = self.df_[self.df_["emotion_id"].notna()].reset_index(drop=True)
        self.df_ = pd.merge(self.df_, self.df_text, on=["smp_id"])
        
    def __load_text__(self, text_path):
        self.df_text = pd.read_csv(text_path)
        pass
    
    def __build_features__(self, raw_list, audio_processer, batch_size=1000):
        self.input_features = []
        last_idx = 0
        for idx in np.arange(batch_size, len(raw_list), batch_size):
            print(idx)
            self.input_features += audio_processer(raw_list[idx-batch_size:idx], 
                                                        sampling_rate=AUDIORATE)['input_features']
            last_idx = idx
        self.input_features += audio_processer(raw_list[last_idx: len(raw_list)], 
                                                    sampling_rate=AUDIORATE)['input_features']
        self.input_features = torch.tensor(self.input_features)
        
    def __load_audio__(self, fn_path, audio_processer):
        self.fn_list = list(self.df_["smp_id"])
        raw_list = []
        for fn in self.fn_list:
            raw_list.append(wavfile.read(os.path.join(fn_path, fn)+'.wav')[1])
        self.__build_features__(raw_list, audio_processer)
    
    def __init__(self, 
                 fn_path, 
                 cutmap_path, 
                 text_path, 
                 pretrained_model="openai/whisper-large"):
        audio_processer = AutoProcessor.from_pretrained(pretrained_model)
        self.available_emotions = {
            "hap": 0,
            "ang": 1,
            "neu": 2,
            "sad": 3,
            "exc": 0,
        }
        self.__load_text__(text_path)
        self.__load_label__(cutmap_path)
        self.__load_audio__(fn_path, audio_processer)
        
    def __len__(self):
        return self.input_features.shape[0]
    
    def __getitem__(self, idx):
        return {
            "smp_id":  self.fn_list[idx],
            "audio": self.input_features[idx],
            "text": self.df_["transcript"][idx],
            "emotion": self.df_["emotion"][idx],
            "labels": self.df_["emotion_id"][idx]
        }