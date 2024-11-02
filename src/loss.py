import torch
from torch import nn
from utils.decodertokens import UNMTDecoderTokens

class ReconstructionLoss:
    def __init__(self, DecoderTokens: UNMTDecoderTokens):
        self.loss = nn.CrossEntropyLoss()
        
    def calculate_loss(self, dec_out: torch.Tensor, target: torch.Tensor):
        # dec_out -> [batch_size, seq_len, vocab_size], [[[0.xx, probabilities...]]]
        # target -> [batch_size, seq_len, index in tokenizer vocab], eg [[[101, 56824, 485, ...]]] 
        # so need to map each index in dec_out to correspoding index of tokenizer vocab & vice versa
        # so then target can be -> [[[1, 5, 3, 8 ...]]]

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

