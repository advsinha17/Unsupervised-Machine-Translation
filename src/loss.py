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
                # if (target[i, j] not in self.decoder_tokens.tokenizer_to_id.keys):
                    # print(target[i, j])
                mapped_target[i, j] = self.decoder_tokens.tokenizer_to_id.get(target[i, j].item(), self.decoder_tokens.tokenizer_to_id[self.tokenizer.unk_token_id])
        
        return mapped_target
        
    def calculate_loss(self, dec_out: torch.Tensor, target: torch.Tensor, attn_mask: torch.Tensor):
        # dec_out -> [batch_size, seq_len, vocab_size], [[[0.xx, probabilities...]]] -> 3d
        # target -> [batch_size, seq_len] of index in tokenizer vocab, eg [[101, 56824, 485, ...]]-> 2d
        # so need to map each index in dec_out to correspoding index of tokenizer vocab & vice versa
        # so then target can be -> [[1, 5, 3, 8 ...]]

        # currently assuming with teacher forcing, pred is of same length as target
        mask = attn_mask.view(-1) == 1
        # print(target)
        target = self.map_target(target)
        dec_out = dec_out.view(-1, dec_out.size(-1))
        target = torch.where(mask, target.view(-1), torch.tensor(self.loss.ignore_index).type_as(target))
        return self.loss(dec_out, target)

class UNMTLoss:
    def __init__(self, src_vocab, tgt_vocab): # TODO: type-hints for src and tgt_vocab
        weight = torch.ones(tgt_vocab) # TODO: get size of tgt_vocab to make 1D tensor
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.loss = nn.NLLLoss(weight, size_average = False)
        
    def calculate_loss(self, src: torch.Tensor, tgt: torch.Tensor):
        loss = 0
        for i in range(tgt.size(1)):
            loss += self.loss(src[i], tgt[i])
        return loss
