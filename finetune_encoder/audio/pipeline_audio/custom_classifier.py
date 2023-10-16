from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn

class CustomClassifier(torch.nn.Module):

    def __init__(self, pretrained_model, num_labels=4):
        super(CustomClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.config = self.encoder.config
        self.config.num_labels = num_labels
        self.projector = nn.Linear(self.config.hidden_size, self.config.classifier_proj_size)
        self.classifier = nn.Linear(self.config.classifier_proj_size, num_labels)

    def forward(
        self,
        input_values,
        attention_mask = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        labels = None,
    ):
        with torch.no_grad():
            outputs = self.encoder(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self.encoder._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        logits = self.classifier(pooled_output)
        return {"logits":logits}