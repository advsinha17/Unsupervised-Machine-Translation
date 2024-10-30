import torch
import torch.nn as nn
from transformers import BertModel


class UNMTEncoder(nn.module):
    def __init__(self, bert_pretrained_type: str = 'bert-base-multilingual-cased', output_dim: int = 768):
        '''
            bert_pretrained_type: str, type of bert model to use
            output_dim: int, dimension of output of encoder
        '''
        super(UNMTEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained_type)
        self.output_dim = output_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        
        x =  self.bert(input_ids, attention_mask = attention_mask)
        x = x.last_hidden_state
        return x # returns (batch, seqlen, 768)