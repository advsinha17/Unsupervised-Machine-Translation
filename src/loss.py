import torch
from torch import nn

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

