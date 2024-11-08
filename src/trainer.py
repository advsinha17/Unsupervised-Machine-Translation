from torch.utils.data import DataLoader
import torch
from torch import nn
from loss import ReconstructionLoss
from utils.decodertokens import UNMTDecoderTokens
from transformers import BertTokenizerFast
from itertools import zip_longest
from tqdm import tqdm
from models.models import SEQ2SEQ

class Trainer:
    def __init__(self, lang1: DataLoader, lang2: DataLoader, lang3: DataLoader, tokenizer: BertTokenizerFast, lr: float = 0.0003): 
        self.lang1 = lang1 # anchor language
        self.lang2 = lang2
        self.lang3 = lang3

        self.decoder_tokens1 = UNMTDecoderTokens(lang1.tokenizer, lang1.lang)
        self.decoder_tokens2 = UNMTDecoderTokens(lang2.tokenizer, lang2.lang)
        self.decoder_tokens3 = UNMTDecoderTokens(lang3.tokenizer, lang3.lang)

        self.reconstruction_loss1 = ReconstructionLoss(self.decoder_tokens1)
        self.reconstruction_loss2 = ReconstructionLoss(self.decoder_tokens2)
        self.reconstruction_loss3 = ReconstructionLoss(self.decoder_tokens3)

        self.tokenizer = tokenizer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.model = None
        self.optimizer = None

    def train(self, model: SEQ2SEQ, epochs: int): 
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = self.lr, betas = (0.5, 0.999))
        self.model = model.to(self.device)
        for epoch in range(epochs):
            model.train()
            max_len = max(len(self.lang1), len(self.lang2), len(self.lang3))
            reconstruction_loss = 0.0
            backtranslation_loss = 0.0
            with tqdm(total = max_len, desc = f"Epoch {epoch + 1} / {epochs}", unit = "batch") as pbar:
                for batch1, batch2, batch3 in zip_longest(self.lang1, self.lang2, self.lang3):
                    self.optimizer.zero_grad()
                    if batch1 is not None:
                        # noisy input 
                        input_ids1, attention_mask1, noisy_input_ids1, noisy_attention_mask1, _, _ = batch1
                        input_ids1 = input_ids1.to(self.device)
                        attention_mask1 = attention_mask1.to(self.device)
                        noisy_input_ids1 = noisy_input_ids1.to(self.device)
                        output_ids1 = self.model(0, input_ids1, None, noisy_input_ids1, None) # check out the 2 nones
                        loss1 = self.reconstruction_loss1.calculate_loss(output_ids1, input_ids1, attention_mask1)
                        loss1.backward()
                        reconstruction_loss += loss1.item()

                        # back translation (0 -> 1 -> 0)
                        with torch.no_grad():
                            bt_out_0_1 = self.model(1, input_ids1, attention_mask1, None, None) 
                        attn_mask_0_1_bt = (bt_out_0_1 != self.tokenizer.pad_token_id).to(self.device)
                        output_ids_0_1_bt = self.model(0, bt_out_0_1, attn_mask_0_1_bt, None, None)
                        loss_0_1_bt = self.reconstruction_loss1.calculate_loss(output_ids_0_1_bt, input_ids1, attention_mask1)
                        loss_0_1_bt.backward()
                        backtranslation_loss += loss_0_1_bt.item()

                        # back translation (0 -> 2 -> 0)    
                        with torch.no_grad():
                            output_ids1_2 = self.model(2, input_ids1, attention_mask1, None, None) 
                        attn_mask1_2_bt = (output_ids1_2 != self.tokenizer.pad_token_id).to(self.device)    
                        output_ids1_2_bt = self.model(0, output_ids1_2, attn_mask1_2_bt, None, None)
                        loss1_2_bt = self.reconstruction_loss1.calculate_loss(output_ids1_2_bt, input_ids1, attention_mask1)
                        loss1_2_bt.backward()
                        backtranslation_loss += loss1_2_bt.item()

                    if batch2 is not None:
                        input_ids2, attention_mask2, noisy_input_ids2, noisy_attention_mask2, _, _ = batch2
                        input_ids2 = input_ids2.to(self.device)
                        attention_mask2 = attention_mask2.to(self.device)
                        noisy_input_ids2 = noisy_input_ids2.to(self.device)
                        _, output_ids2, _ = self.model(noisy_input_ids2, attention_mask2)
                        loss2 = self.reconstruction_loss2.calculate_loss(output_ids2, input_ids2, attention_mask2)
                        loss2.backward()
                        reconstruction_loss += loss2.item()

                        # back translation (1 -> 0 -> 1)
                        with torch.no_grad():
                            bt_out_1_0 = self.model(0, input_ids1, attention_mask1, None, None) 
                        attn_mask_1_0 = (bt_out_1_0 != self.tokenizer.pad_token_id).to(self.device)
                        output_ids_1_0_bt = self.model(1, bt_out_1_0, attn_mask_1_0, None, None)
                        loss_1_0_bt = self.reconstruction_loss2.calculate_loss(output_ids_1_0_bt, input_ids2, attention_mask2) 
                        loss_1_0_bt.backward()
                        backtranslation_loss += loss_1_0_bt.item()  
                    
                    if batch3 is not None:
                        input_ids3, attention_mask3, noisy_input_ids3, noisy_attention_mask3, _, _ = batch3
                        input_ids3 = input_ids3.to(self.device)
                        attention_mask3 = attention_mask3.to(self.device)
                        noisy_input_ids3 = noisy_input_ids3.to(self.device)
                        _, _, output_ids3 = self.model(noisy_input_ids3, attention_mask3)
                        loss3 = self.reconstruction_loss3.calculate_loss(output_ids3, input_ids3, attention_mask3)
                        loss3.backward()
                        reconstruction_loss += loss3.item()
                        with torch.no_grad():
                            bt_out_2_0 = self.model(0, input_ids1, attention_mask1, None, None) # maybe replace None with noisy input mask
                        attn_mask_2_0 = (bt_out_2_0 != self.tokenizer.pad_token_id).to(self.device)
                        output_ids_2_0_bt = self.model(1, bt_out_2_0, attn_mask_2_0, None, None)
                        loss_2_0_bt = self.reconstruction_loss3.calculate_loss(output_ids_2_0_bt, input_ids3, attention_mask3) 
                        loss_2_0_bt.backward()
                        backtranslation_loss += loss_2_0_bt.item() 

                    self.optimizer.step()
                    pbar.update(1)
                

            print(f"Epoch {epoch + 1} / {epochs}, Reconstruction Loss: {reconstruction_loss}, Backtranslation Loss: {backtranslation_loss}")












