import torch
import torch.nn as nn

def forward_padding_mask(
        features: torch.Tensor, padding_mask: torch.Tensor,
    ):
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask
    
class AVHUBERTClassifier(torch.nn.Module):

    def __init__(self, encoder, hidden_size, proj_size, num_labels=4):
        super(AVHUBERTClassifier, self).__init__()
        self.encoder = encoder
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.num_labels = num_labels
        
        self.projector = nn.Linear(self.hidden_size, self.proj_size)
        self.classifier = nn.Linear(self.proj_size, num_labels)

    def forward(
        self,
        audio, video, padding_mask, **kwargs
    ):
        with torch.no_grad():
            outputs = self.encoder.extract_finetune(
                {
                    "audio": audio, 
                    "video": video,
                    "padding_mask": padding_mask
                }
            )
            hidden_states = outputs[0]
        
        hidden_states = self.projector(hidden_states)
        attention_mask = padding_mask
        padding_mask = forward_padding_mask(hidden_states, attention_mask)
        hidden_states[~padding_mask] = 0.0
        pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        logits = self.classifier(pooled_output)
        return {"logits":logits}
    