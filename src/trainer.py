from torch.utils.data import DataLoader
import torch
from torch import nn
from .loss import ReconstructionLoss
from .utils.decodertokens import UNMTDecoderTokens
from transformers import XLMRobertaTokenizerFast
from itertools import zip_longest
from tqdm import tqdm
from .models.models import SEQ2SEQ

class Trainer:
    def __init__(self, langs: list, tokenizer: XLMRobertaTokenizerFast, lr: float = 0.0003): 

        self.n_langs = len(langs)

        for i in range(self.n_langs):
            setattr(self, f"lang{i + 1}", langs[i])
            setattr(self, f"decoder_tokens{i + 1}", UNMTDecoderTokens(langs[i].dataset.tokenizer, langs[i].dataset.lang))
            getattr(self, f"decoder_tokens{i + 1}").load_token_list()
            setattr(self, f"reconstruction_loss{i + 1}", ReconstructionLoss(getattr(self, f"decoder_tokens{i + 1}")))

        self.tokenizer = tokenizer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.model = None
        self.optimizer = None

    def map_argmax_to_tokenizer_id(self, argmax_tensor, decoder_tokens):
        mapped_tensor = argmax_tensor.clone()
        for i in range(argmax_tensor.size(0)):
            for j in range(argmax_tensor.size(1)):
                mapped_tensor[i, j] = decoder_tokens.id_to_tokenizer.get(argmax_tensor[i, j].item(), decoder_tokens.tokenizer.pad_token_id)
        return mapped_tensor

    def train(self, model: SEQ2SEQ, epochs: int): 
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = self.lr, betas = (0.5, 0.999))
        self.model = model.to(self.device)
        for epoch in range(epochs):
            model.train()
            max_len = max(len(getattr(self, f"lang{i + 1}")) for i in range(self.n_langs))
            reconstruction_loss = 0.0
            backtranslation_loss = 0.0
            with tqdm(total = max_len, desc = f"Epoch {epoch + 1} / {epochs}", unit = "batch") as pbar:
                for batches in zip_longest(*(getattr(self, f"lang{i + 1}") for i in range(self.n_langs))):

                    self.optimizer.zero_grad()
                    if batches[0] is not None:
                        # noisy input 
                        input_ids1, attention_mask1, noisy_input_ids1, noisy_attention_mask1, _, _ = batches[0]
                        input_ids1 = input_ids1.to(self.device)
                        attention_mask1 = attention_mask1.to(self.device)
                        noisy_input_ids1 = noisy_input_ids1.to(self.device)
                        noisy_attention_mask1 = noisy_attention_mask1.to(self.device)
                        output_ids1 = self.model(0, input_ids1, attention_mask1, noisy_input_ids1, noisy_attention_mask1, True, True) 
                        loss1 = getattr(self, "reconstruction_loss1").calculate_loss(output_ids1, input_ids1, attention_mask1)
                        loss1.backward()
                        reconstruction_loss += loss1.item()

                        # back translation (0 -> 1 -> 0)
                        with torch.no_grad():
                            bt_out_0_1 = self.model(1, input_ids1, attention_mask1, None, None, False, False) 
                            bt_out_0_1 = torch.argmax(bt_out_0_1, dim=-1)
                            bt_out_0_1 = self.map_argmax_to_tokenizer_id(bt_out_0_1, self.decoder_tokens2)
                        attn_mask_0_1_bt = (bt_out_0_1 != self.tokenizer.pad_token_id).to(self.device)
                        output_ids_0_1_bt = self.model(0, input_ids1, attention_mask1, bt_out_0_1, attn_mask_0_1_bt, True, True)
                        loss_0_1_bt = getattr(self, "reconstruction_loss1").calculate_loss(output_ids_0_1_bt, input_ids1, attention_mask1)
                        loss_0_1_bt.backward()
                        backtranslation_loss += loss_0_1_bt.item()

                        # back translation (0 -> 2 -> 0)    
                        with torch.no_grad():
                            bt_out_0_2 = self.model(2, input_ids1, attention_mask1, None, None, False, False)
                            bt_out_0_2 = torch.argmax(bt_out_0_2, dim=-1) 
                            bt_out_0_2 = self.map_argmax_to_tokenizer_id(bt_out_0_2, self.decoder_tokens3)
                        attn_mask_0_2_bt = (bt_out_0_2 != self.tokenizer.pad_token_id).to(self.device)    
                        output_ids_0_2_bt = self.model(0, input_ids1, attention_mask1, bt_out_0_2, attn_mask_0_2_bt, True, True)
                        loss_0_2_bt = getattr(self, "reconstruction_loss1").calculate_loss(output_ids_0_2_bt, input_ids1, attention_mask1)
                        loss_0_2_bt.backward()
                        backtranslation_loss += loss_0_2_bt.item()

                    for i in range(1, self.n_langs):
                        if batches[i] is not None:
                            input_ids, attention_mask, noisy_input_ids, noisy_attention_mask, _, _ = batches[i]
                            input_ids = input_ids.to(self.device)
                            attention_mask = attention_mask.to(self.device)
                            noisy_input_ids = noisy_input_ids.to(self.device)
                            noisy_attention_mask = noisy_attention_mask.to(self.device)
                            output_ids = self.model(i, input_ids, attention_mask, noisy_input_ids, noisy_attention_mask, True, True)
                            loss = getattr(self, f"reconstruction_loss{i + 1}").calculate_loss(output_ids, input_ids, attention_mask)
                            loss.backward()
                            reconstruction_loss += loss.item()

                            with torch.no_grad():
                                bt_out = self.model(0, input_ids, attention_mask, None, None, False, False)
                                bt_out = torch.argmax(bt_out, dim=-1)
                                bt_out = self.map_argmax_to_tokenizer_id(bt_out, getattr(self, f"decoder_tokens{i + 1}"))
                            attn_mask_bt = (bt_out != self.tokenizer.pad_token_id).to(self.device)
                            output_ids_bt = self.model(i, input_ids, attention_mask, bt_out, attn_mask_bt, True, True)
                            loss_bt = getattr(self, f"reconstruction_loss1").calculate_loss(output_ids_bt, input_ids, attention_mask)
                            loss_bt.backward()
                            backtranslation_loss += loss_bt.item()

                    self.optimizer.step()
                    pbar.update(1)
                

            print(f"Epoch {epoch + 1} / {epochs}, Reconstruction Loss: {reconstruction_loss}, Backtranslation Loss: {backtranslation_loss}")
            torch.save(self.model.state_dict(), 'model.pth')












