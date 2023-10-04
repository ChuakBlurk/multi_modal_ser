import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MMSERDataset(Dataset):
    """multi model ser dataset."""
    
    def __load_audio__(self, fn_path):
        # load all spectrogram
        self.fn_list = list(self.df_["smp_id"])
        self.raw_dict = {}
        for fn in self.fn_list[:1000]:
            self.raw_dict[fn] = wavfile.read(os.path.join(fn_path, fn)+'.wav')[1]
        self.raw_list = list(self.raw_dict.values())
        
    def __load_label__(self, cutmap_path):
        sheet_df = pd.DataFrame()
        for ses in range(1, 6):
            extractionmapPATH = cutmap_path + \
                str(ses)+'.xlsx'
            xl = pd.ExcelFile(extractionmapPATH)
            sheets = xl.sheet_names
            for sheet in sheets:
                sheet_df = pd.concat([sheet_df, xl.parse(sheet)])
        self.df_ = sheet_df.reset_index(drop=True)
        self.df_ = pd.merge(self.df_, self.df_text, on=["smp_id"])
        self.labels = list(self.df_["emotion"].unique())
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.df_["emotion_id"] = self.df_["emotion"].map(self.label2id)
        
    def __load_text__(self, text_path):
        self.df_text = pd.read_csv(text_path)
        pass
    
    def __build_features__(self):
        self.input_features = self.audio_processer(self.raw_list, sampling_rate=AUDIORATE, return_tensors="pt")['input_features']
    
    def __init__(self, 
                 fn_path, 
                 cutmap_path, 
                 text_path, 
                 pretrained_model="openai/whisper-base.en"):
        self.audio_processer = AutoProcessor.from_pretrained(pretrained_model)
#         self.audio_encoder = WhisperForConditionalGeneration.from_pretrained(pretrained_model)
        self.__load_text__(text_path)
        self.__load_label__(cutmap_path)
        self.__load_audio__(fn_path)
        self.__build_features__()
        
    def __len__(self):
        return self.input_features.shape[0]
    
    def __getitem__(self, idx):
        return {
#             "fn": self.fn_list[idx],
#             "raw": self.raw_list[idx],
            "input_features": self.input_features[idx],
#             "text": self.df_["transcript"][idx],
            "labels": self.df_["emotion_id"][idx]
        }