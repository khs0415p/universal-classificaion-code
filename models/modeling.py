import torch.nn as nn

from transformers import AutoModelForSequenceClassification


class ClassificationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_path, num_labels=config.num_labels, cache_dir=config.cache_dir, trust_remote_code=True)
        self.config = self.model.config


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
            }

        if self.config.model_type in ["roberta", "distilbert"]:
            kwargs.pop("token_type_ids")

        output = self.model(**kwargs)
        return output
    


