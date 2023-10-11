from datasets import load_metric
from transformers import AutoFeatureExtractor, WhisperForAudioClassification
from datasets import load_metric
import numpy as np
from transformers import Trainer
from transformers import TrainingArguments
import datetime
import sys
from torch import nn
sys.path.append("E:/university/Year 5 Spring/FYT/code/multi_modal_ser")
from utils.dataset import MMSERDataset
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mmser_ds = torch.load("E:/datasets/preprocessed/dataset/mmser_ds.pt")
train_size = int(len(mmser_ds)*0.7)
val_size = int(len(mmser_ds)*0.2)
test_size = len(mmser_ds)-int(len(mmser_ds)*0.7)-int(len(mmser_ds)*0.2)

train_set, val_set = torch.utils.data.random_split(mmser_ds, [train_size, val_size+test_size])
val_set, test_set = torch.utils.data.random_split(val_set, [val_size, test_size])
smallval_set, _ = torch.utils.data.random_split(val_set, [10, val_size-10])

def compute_metrics(eval_preds):
    print(eval_preds)
    metric_f1 = load_metric("f1")
    metric_acc = load_metric("accuracy")
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    predictions = np.argmax(logits, axis=-1)
    f1_ = metric_f1.compute(predictions=predictions, references=labels, average="weighted")
    acc_ = metric_acc.compute(predictions=predictions, references=labels)
    return {"F1":f1_, "acc":acc_}

output_dir=datetime.datetime.now().date().strftime(format="%Y-%m-%d")
training_args = TrainingArguments(output_dir)
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        outputs = model(input_features=inputs["audio"].to(device))
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss() # weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

training_args.remove_unused_columns=False
model = WhisperForAudioClassification.from_pretrained("E:/university/Year 5 Spring/FYT/code/multi_modal_ser/finetune_encoder/finetune/2023-10-07")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=smallval_set,
    compute_metrics=compute_metrics,
)

trainer.evaluate()