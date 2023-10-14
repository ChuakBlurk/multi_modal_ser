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
    
    def __init__(self, 
                 fn_path, 
                 cutmap_path, 
                 text_path, 
                 emo2id):
        self.emo2id = emo2id
        self.__load_text__(text_path)
        self.__load_label__(cutmap_path)
        self.__load_audio__(fn_path)
        
    def __len__(self):
        return self.df_.shape[0]
    
    def __getitem__(self, idx):
        return {
            "sess": list(self.df_["session"])[idx],
            "fn": self.fn_list[idx],
            "audio": self.raw_list[idx],
            "text": list(self.df_["transcript"])[idx],
            "labels": list(self.df_["emotion_id"])[idx]
        }