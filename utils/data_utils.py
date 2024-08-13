import json
import torch
import pandas as pd

from typing import List
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, config, data_path, tokenizer):
        self.config = config

        self.max_length = config.max_length
        self.tokenizer = tokenizer

        self.data = pd.read_csv(data_path)

        with open(config.label2id, "r") as f:
            self.label2id = json.load(f)


    def make_data(self, sentence: str):
        output = self.tokenizer(
            sentence,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        return output

    def __getitem__(self, index):
        row = self.data.iloc[index]

        label = self.label2id[row['label']]
        tokenized_sentence = self.make_data(row['content'])

        return {
            "input_ids" : tokenized_sentence.input_ids.squeeze(),
            "token_type_ids": tokenized_sentence.token_type_ids.squeeze() if self.config.model_type != "roberta" else None,
            "labels": torch.LongTensor([label])
        }


    def __len__(self):
        return len(self.data)