from transformers import AutoProcessor, HubertModel
from torch.utils.data import Dataset, Subset
from torch.utils.data import Dataset, DataLoader
import sys
# sys.path.append("E:/university/FYT/repos/multi_modal_ser")
sys.path.append("/home/multi_modal_ser")
from tqdm import tqdm



class ProcessedDataset(Dataset):
    
    def __init__(self, base_ds, pretrained_model):
        self.base_ds = base_ds
        self.processor = AutoProcessor.from_pretrained(pretrained_model)
        self.__process__()
        
    def __process__(self):
        self.input_values_list = []
        self.attention_mask_list = []
        for raw_audio in tqdm(self.base_ds.raw_list):
            processed = self.processor(raw_audio, 
                               sampling_rate=16000,
                               padding='max_length',
                               max_length=300000,
                               truncation=True, 
                              return_tensors="np")
            self.input_values_list.append(processed["input_values"].squeeze())
            self.attention_mask_list.append(processed["attention_mask"].squeeze())
    def __len__(self):
        return len(self.base_ds)
    
    def __getitem__(self, idx):
        base_dict = self.base_ds[idx]
        base_dict["input_values"] = self.input_values_list[idx]
        base_dict["attention_mask"] = self.attention_mask_list[idx]
        del base_dict["audio"]
        return base_dict
        
        