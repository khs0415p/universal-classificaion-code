import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTrainer
from utils.train_utils import get_dataloader, get_test_dataloader
from transformers import AutoTokenizer



class Trainer(BaseTrainer):
    def __init__(self, config, device) -> None:
        super().__init__(config, device)

        # dataloaders
        if config.mode == "train":
            self.dataloader, self.tokenizer = get_dataloader(config) # {'train': dataloader, 'valid': dataloader}
            self.config.vocab_size = len(self.tokenizer) 

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint, trust_remote_code=True, cache_dir=self.config.cache_dir)
            self.dataloader = get_test_dataloader(config, self.tokenizer)

        # acc history
        self.valid_acc_history = []

        # main process
        self.rank_zero = True if not self.ddp or (self.ddp and device == 0) else False

        # initialize trainer
        self._init_trainer()

        # criterion
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)


    def focal_loss(self, loss):

        pt = torch.exp(-loss)
        focal_loss = self.config.alpha * (1-pt)**self.config.gamma * loss
        focal_loss = torch.mean(focal_loss)
        return focal_loss


    def _training_step(self, model_inputs):
        """
        Args:
            model_inputs: data of batch
        Return:
            (Tensor loss): loss
        """        
        
        output = self.model(
            input_ids=model_inputs['input_ids'],    # batch, seq
            attention_mask=model_inputs['attention_mask'],
            token_type_ids=model_inputs['token_type_ids'] if 'token_type_ids' in model_inputs else None
        )
        
        logits = output.logits # batch, num_labels
        loss = self.cross_entropy(logits.view(-1, self.config.num_labels), model_inputs['labels'].view(-1))
        loss = self.focal_loss(loss)

        self._backward_step(loss)

        return loss.item()


    @torch.no_grad()
    def _validation_step(self, model_inputs):
        """
        Args:
            model_inputs: data of batch
        Return:
            (Tensor loss): loss
        """
        output = self.model(
            input_ids=model_inputs['input_ids'],    # batch, seq
            attention_mask=model_inputs['attention_mask'],
            token_type_ids=model_inputs['token_type_ids'] if 'token_type_ids' in model_inputs else None
        )
        
        logits = output.logits # batch, num_labels
        loss = self.cross_entropy(logits.view(-1, self.config.num_labels), model_inputs['labels'].view(-1))
        loss = self.focal_loss(loss)

        outputs = torch.argmax(logits.detach().cpu(), dim=-1)
        acc = torch.sum(outputs == model_inputs['labels'].detach().cpu().squeeze()) / logits.size(0)

        return loss.item(), acc.item(), logits

    @torch.no_grad()
    def _test_step(self, model_inputs):

        output = self.model(
            input_ids=model_inputs['input_ids'],    # batch, seq
            attention_mask=model_inputs['attention_mask'],
            token_type_ids=model_inputs['token_type_ids'] if 'token_type_ids' in model_inputs else None
        )
        
        logits = output.logits # batch, num_labels
        pred = torch.argmax(logits.detach().cpu(), dim=-1)
        label = model_inputs['labels'].detach().cpu().squeeze()
        

        return pred.tolist(), label.tolist()