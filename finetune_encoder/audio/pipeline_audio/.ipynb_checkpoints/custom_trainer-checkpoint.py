import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, WhisperForAudioClassification, Trainer, TrainingArguments
import numpy as np
from datasets import load_metric


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").type(torch.LongTensor).to(device)
        
        input_values = inputs["input_values"].to(device).to(torch.float32)
        attention_mask = inputs["attention_mask"].to(device).to(torch.float32)
        outputs = model(input_values, 
                       attention_mask)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss() 
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))        
        return (loss, outputs) if return_outputs else loss
    
def weighted_acc(y_true, y_pred):
    return np.sum((np.array(y_pred).ravel() == np.array(y_true).ravel()))*1.0/len(y_true)
    
def unweighted_acc(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    classes = np.unique(y_true)
    classes_accuracies = np.zeros(classes.shape[0])
    for num, cls in enumerate(classes):
        classes_accuracies[num] = weighted_acc(y_true[y_true == cls], y_pred[y_true == cls])
    return np.mean(classes_accuracies)

def compute_metrics(eval_preds):
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    predictions = np.argmax(logits, axis=-1)

    metric_f1 = load_metric("f1")
    metric_acc = load_metric("accuracy")
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    predictions = np.argmax(logits, axis=-1)
    f1_ = metric_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    acc_ = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    
    return {"wa":weighted_acc(labels, predictions), 
            "ua":unweighted_acc(labels, predictions),
            "f1":f1_, 
            "accuracy":acc_}