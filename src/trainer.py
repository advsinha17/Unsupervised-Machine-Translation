from torch.utils.data import DataLoader
import torch
from torch import nn
from loss import ReconstructionLoss
from utils.decodertokens import UNMTDecoderTokens
from itertools import zip_longest
from tqdm import tqdm

class Trainer:
    def __init__(self, lang1: DataLoader, lang2: DataLoader, lang3: DataLoader, lr: float = 0.0003): # TODO: model type-hint
        self.lang1 = lang1 # anchor language
        self.lang2 = lang2
        self.lang3 = lang3

        self.decoder_tokens1 = UNMTDecoderTokens(lang1.tokenizer, lang1.lang)
        self.decoder_tokens2 = UNMTDecoderTokens(lang2.tokenizer, lang2.lang)
        self.decoder_tokens3 = UNMTDecoderTokens(lang3.tokenizer, lang3.lang)

        self.reconstruction_loss1 = ReconstructionLoss(self.decoder_tokens1)
        self.reconstruction_loss2 = ReconstructionLoss(self.decoder_tokens2)
        self.reconstruction_loss3 = ReconstructionLoss(self.decoder_tokens3)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.model = None
        self.optimizer = None

    def train(self, model, epochs: int): # TODO: model type-hint
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = self.lr, betas = (0.5, 0.999))
        self.model = model.to(self.device)
        for epoch in range(epochs):
            model.train()
            max_len = max(len(self.lang1), len(self.lang2), len(self.lang3))
            reconstruction_loss = 0.0
            with tqdm(total = max_len, desc = f"Epoch {epoch + 1} / {epochs}", unit = "batch") as pbar:
                for batch1, batch2, batch3 in zip_longest(self.lang1, self.lang2, self.lang3):
                    self.optimizer.zero_grad()
                    if batch1 is not None:
                        input_ids1, attention_mask1, noisy_input_ids1, _, _ = batch1
                        input_ids1 = input_ids1.to(self.device)
                        attention_mask1 = attention_mask1.to(self.device)
                        noisy_input_ids1 = noisy_input_ids1.to(self.device)
                        output_ids1, _, _ = self.model(noisy_input_ids1, attention_mask1)
                        loss1 = self.reconstruction_loss1.calculate_loss(output_ids1, input_ids1)
                        loss1.backward()
                        reconstruction_loss += loss1.item()

                    if batch2 is not None:
                        input_ids2, attention_mask2, noisy_input_ids2, _, _ = batch2
                        input_ids2 = input_ids2.to(self.device)
                        attention_mask2 = attention_mask2.to(self.device)
                        noisy_input_ids2 = noisy_input_ids2.to(self.device)
                        output_ids2, _, _ = self.model(noisy_input_ids2, attention_mask2)
                        loss2 = self.reconstruction_loss2.calculate_loss(output_ids2, input_ids2)
                        loss2.backward()
                        reconstruction_loss += loss2.item()
                    
                    if batch3 is not None:
                        input_ids3, attention_mask3, noisy_input_ids3, _, _ = batch3
                        input_ids3 = input_ids3.to(self.device)
                        attention_mask3 = attention_mask3.to(self.device)
                        noisy_input_ids3 = noisy_input_ids3.to(self.device)
                        output_ids3, _, _ = self.model(noisy_input_ids3, attention_mask3)
                        loss3 = self.reconstruction_loss3.calculate_loss(output_ids3, input_ids3)
                        loss3.backward()
                        reconstruction_loss += loss3.item()

                    self.optimizer.step()
                    pbar.update(1)
                    
            print(f"Epoch {epoch + 1} / {epochs}, Reconstruction Loss: {reconstruction_loss}")











