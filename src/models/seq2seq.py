import torch
import torch.nn as nn
from transformers import BertTokenizerFast
from src.utils.decodertokens import UNMTDecoderTokens
from Encoder import UNMTEncoder
from Decoder import LSTM_ATTN_Decoder


class SEQ2SEQ(nn.Module):
    def __init__(self, tokenizer: BertTokenizerFast, bert_pretrained_type:str = 'bert-base-multilingual-cased', mode:str = 'Train',
                 decoder_hidden_dim:int = 768):
        super(SEQ2SEQ, self).__init__()
        self.mode = mode

        self.encoder = UNMTEncoder(bert_pretrained_type = bert_pretrained_type)
        self.embedding_layer = self.encoder.bert.embeddings.word_embeddings

        self.decoder_en = LSTM_ATTN_Decoder(lang = 'en', tokenizer = tokenizer, 
                                     embedding_layer = self.embedding_layer, mode=self.mode,
                                     hidden_dim = decoder_hidden_dim)
        self.decoder_hi = LSTM_ATTN_Decoder(lang = 'hi', tokenizer = tokenizer, 
                                     embedding_layer = self.embedding_layer, mode=self.mode,
                                     hidden_dim = decoder_hidden_dim)
        self.decoder_te = LSTM_ATTN_Decoder(lang = 'te', tokenizer = tokenizer,
                                     embedding_layer = self.embedding_layer, mode=self.mode,
                                     hidden_dim = decoder_hidden_dim)
        
        
        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, noisy_input_ids: torch.Tensor = None, noisy_attention_mask = None, 
                    g_truth: bool = False):
            if self.mode == 'Train':
                assert noisy_input_ids is not None, "noisy_input_ids must be provided in Train mode"
                assert noisy_attention_mask is not None, "noisy_attention_mask must be provided in Train mode"

                #so the noisy is actually the input to the encoder in this case.

                encoder_output = self.encoder(noisy_input_ids, attention_mask = noisy_attention_mask)

                decoder_en_output = self.decoder_en(encoder_output, target_seq = input_ids, g_truth = g_truth)
                decoder_hi_output = self.decoder_hi(encoder_output, target_seq = input_ids, g_truth = g_truth)
