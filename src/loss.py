import torch
from torch import nn
from .utils.decodertokens import UNMTDecoderTokens
from transformers import XLMRobertaTokenizerFast

class ReconstructionLoss:
    def __init__(self, DecoderTokens: UNMTDecoderTokens):
        self.loss = nn.CrossEntropyLoss()
        self.decoder_tokens = DecoderTokens
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')

    def map_target(self, target: torch.Tensor):
        mapped_target = target.clone()
        for i in range(target.size(0)):
            for j in range(target.size(1)):
                mapped_target[i, j] = self.decoder_tokens.tokenizer_to_id.get(target[i, j].item(), self.decoder_tokens.tokenizer_to_id[self.tokenizer.unk_token_id])
        
        return mapped_target
        
    def calculate_loss(self, dec_out: torch.Tensor, target: torch.Tensor, attn_mask: torch.Tensor):
        # dec_out -> [batch_size, seq_len, vocab_size], [[[0.xx, probabilities...]]] -> 3d
        # target -> [batch_size, seq_len] of index in tokenizer vocab, eg [[101, 56824, 485, ...]]-> 2d
        # so need to map each index in dec_out to correspoding index of tokenizer vocab & vice versa
        # so then target can be -> [[1, 5, 3, 8 ...]]

        mask = attn_mask.view(-1) == 1
        target = self.map_target(target)
        dec_out = dec_out.view(-1, dec_out.size(-1))
        target = torch.where(mask, target.view(-1), torch.tensor(self.loss.ignore_index).type_as(target))
        return self.loss(dec_out, target)

