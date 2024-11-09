import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel
from src.utils.decodertokens import UNMTDecoderTokens

class UNMTEncoder(nn.Module):
    def __init__(self, bert_pretrained_type: str = 'bert-base-multilingual-cased'):
        '''
            bert_pretrained_type: str, type of bert model to use
            output_dim: int, dimension of output of encoder
        '''
        super(UNMTEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained_type)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        
        x =  self.bert(input_ids, attention_mask = attention_mask)
        x = x.last_hidden_state
        return x # returns (batch, seqlen, 768)

class LSTM_ATTN_Decoder(nn.Module):
    def __init__(self, lang:str , tokenizer: BertTokenizerFast, embedding_layer: torch.nn.modules.sparse.Embedding,
                 mode:str = 'Train',
                 max_output_length:int = 500 , dropout_rate:float = 0.3 ,num_layers:int = 3,
                 embedding_dim: int = 768, hidden_dim:int = 1024):
        
        assert lang in ['en', 'te', 'hi'], "lang must be one of 'en', 'te', or 'hi'"

        super(LSTM_ATTN_Decoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding_layer = embedding_layer

        self.mode = mode # either 'Train' or 'Test'
        self.lang = lang
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.tokenizer = tokenizer

        self.start_token_id = self.tokenizer.cls_token_id
        self.end_token_id = self.tokenizer.sep_token_id

        self.decoderTokens = UNMTDecoderTokens(tokenizer = self.tokenizer, lang = lang)
        self.decoderTokens.load_token_list()
        self.decoderTokens.token_set = torch.tensor(self.decoderTokens.token_set).to(self.device) #1D tensor of valid tokens
        
        
        self.no_output_tokens = self.decoderTokens.token_set.size(dim = 0)
        self.max_output_length = max_output_length #maximum length of output sequence
        

        self.LSTM = nn.LSTM(input_size=2*self.embedding_dim,
                            hidden_size=self.hidden_dim, 
                            num_layers=self.num_layers, 
                            dropout=self.dropout_rate,
                            batch_first=True)
        
        self.Attention_MLP = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # outputs a single score for attention
        )
        
        self.fc_final = nn.Linear(hidden_dim, self.no_output_tokens) 
        self.relu = nn.ReLU()
        self.softmax_layer = nn.LogSoftmax(dim = 1)

    def forward(self, x:torch.Tensor, target_seq:torch.Tensor = None, g_truth:bool = False):
        '''
            x: (batch_size, seq_len1, embedding_dim)
            target_seq: (batch_size, seq_len_target)
            g_truth: bool, if True, then teacher forcing is used
        '''
        batch_size = x.size(0)
        seq_len1 = x.size(1)
        prev_pred_tokens = None #initially None and later gets updated with the previous predicted token values
        # prev_pred_token shape is [batch_size]

        #initializing LSTM hidden and cell states
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim ).to(self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim ).to(self.device)

        outputs = []

        if self.mode == 'Train':
            
            assert target_seq is not None, "target_seq must be provided in Train mode"
            
            seq_len_target = target_seq.size(1)
        
            assert seq_len_target <= self.max_output_length, "seq_len_target must be less than or equal to max_output_length"

            for t in range(seq_len_target):
                
                attn_scores = []
                for i in range(seq_len1):
                    
                    attn_input = torch.cat((x[:, i, :], h[-1]), dim = 1).to(self.device)  # h[-1] is the hidden state from last layer
                    #attn_input has the shape [batch, embedding_dim + hidden_dim]

                    attn_scores.append(self.Attention_MLP(attn_input)) # seq_len1 no of elements appended to the list; of shape [batch_size, 1]

                attn_scores_tensor = torch.stack(attn_scores, dim = 1).to(self.device) #[batch_size, seq_len1, 1]
                attn_scores_tensor = self.softmax_layer(attn_scores_tensor).to(self.device)

                context_vector = torch.sum(attn_scores_tensor * x, dim = 1).to(self.device) #[batch_size, embedding_dim]

                if t==0 or g_truth:
                    current_intput_token = target_seq[:, t] #[batch_size]
                    current_input = self.embedding_layer(current_intput_token).to(self.device) #[batch_size, embedding_dim]
                else:
                    current_input = self.embedding_layer(prev_pred_tokens).to(self.device)  #[batch_size, embedding_dim]

                lstm_input = torch.cat((current_input, context_vector), dim = 1).unsqueeze(1) #[batch_size, 1 ,2*embedding_dim]

                lstm_output, (h,c) = self.LSTM(lstm_input, (h,c))

                pred_prob = self.fc_final(lstm_output.squeeze(1)) #[batch_size, no_output_tokens]
                pred_prob_softmax = self.softmax_layer(pred_prob)

                prev_pred_token_indices = pred_prob_softmax.argmax(dim = 1) #[batch_size]
                prev_pred_tokens = self.decoderTokens.token_set[prev_pred_token_indices] #[batch_size, embedding_dim]

                outputs.append(pred_prob_softmax)

        elif self.mode == 'Test':

            for t in range(self.max_output_length):
                
                attn_scores = []
                for i in range(seq_len1):
                    
                    attn_input = torch.cat((x[:, i, :], h[-1]), dim = 1)  # h[-1] is the hidden state from last layer
                    #attn_input has the shape [batch, embedding_dim + hidden_dim]

                    attn_scores.append(self.Attention_MLP(attn_input)) # seq_len1 no of elements appended to the list; of shape [batch_size, 1]

                attn_scores_tensor = torch.stack(attn_scores, dim = 1) #[batch_size, seq_len1, 1]
                attn_scores_tensor = self.softmax_layer(attn_scores_tensor)

                context_vector = torch.sum(attn_scores_tensor * x, dim = 1) #[batch_size, embedding_dim]

                if t==0:
                    current_intput_token =torch.tensor([self.start_token_id]*batch_size).to(self.device) #[batch_size]
                    current_input = self.embedding_layer(current_intput_token).to(self.device) #[batch_size, embedding_dim]
                else:
                    current_input = self.embedding_layer(prev_pred_tokens).to(self.device)  #[batch_size, embedding_dim]

                lstm_input = torch.cat((current_input, context_vector), dim = 1).unsqueeze(1) #[batch_size, 1 ,2*embedding_dim]

                lstm_output, (h,c) = self.LSTM(lstm_input, (h,c))

                pred_prob = self.fc_final(lstm_output.squeeze(1)) #[batch_size, no_output_tokens]
                pred_prob_softmax = self.softmax_layer(pred_prob)

                prev_pred_token_indices = pred_prob_softmax.argmax(dim = 1) #[batch_size]
                prev_pred_tokens = self.decoderTokens.token_set[prev_pred_token_indices] #[batch_size, embedding_dim]

                outputs.append(pred_prob_softmax)

                if(prev_pred_tokens == self.end_token_id).all():
                    break
        
        outputs = torch.stack(outputs, dim = 1)

        return outputs

class SEQ2SEQ(nn.Module):
    def __init__(self, tokenizer: BertTokenizerFast, bert_pretrained_type:str = 'bert-base-multilingual-cased', mode:str = 'Train',
                 decoder_hidden_dim:int = 768):
        super(SEQ2SEQ, self).__init__()
        self.mode = mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = UNMTEncoder(bert_pretrained_type = bert_pretrained_type)
        self.embedding_layer = self.encoder.bert.embeddings.word_embeddings.to(self.device)

        self.decoder_en = LSTM_ATTN_Decoder(lang = 'en', tokenizer = tokenizer, 
                                     embedding_layer = self.embedding_layer, mode=self.mode,
                                     hidden_dim = decoder_hidden_dim)
        self.decoder_hi = LSTM_ATTN_Decoder(lang = 'hi', tokenizer = tokenizer, 
                                     embedding_layer = self.embedding_layer, mode=self.mode,
                                     hidden_dim = decoder_hidden_dim)
        self.decoder_te = LSTM_ATTN_Decoder(lang = 'te', tokenizer = tokenizer,
                                     embedding_layer = self.embedding_layer, mode=self.mode,
                                     hidden_dim = decoder_hidden_dim)
        
        
    def forward(self, language:int ,
                input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                noisy_input_ids: torch.Tensor = None, noisy_attention_mask = None, 
                g_truth: bool = False, noisy_input: bool = False):
        #language is 0 for en, 1 for hi and 2 for te
        if self.mode == 'Train':
            assert noisy_input_ids is not None, "noisy_input_ids must be provided in Train mode"
            assert noisy_attention_mask is not None, "noisy_attention_mask must be provided in Train mode"

            #so the noisy is actually the input to the encoder in this case.
            #but we dont want noise for backtranslation so we have a flag for that
            if(noisy_input):
                encoder_output = self.encoder(noisy_input_ids, attention_mask = noisy_attention_mask)
            else:
                encoder_output = self.encoder(input_ids, attention_mask = attention_mask)
            
            if language == 0:
                decoder_output = self.decoder_en(encoder_output, target_seq = input_ids, g_truth = g_truth)
            elif language ==1:
                decoder_output = self.decoder_hi(encoder_output, target_seq = input_ids, g_truth = g_truth)
            else:
                decoder_output = self.decoder_te(encoder_output, target_seq = input_ids, g_truth = g_truth)

            return decoder_output
        
        if self.mode == 'Test':
            encoder_output = self.encoder(input_ids, attention_mask = attention_mask)
            if language == 0:
                decoder_output = self.decoder_en(encoder_output)
            elif language ==1:
                decoder_output = self.decoder_hi(encoder_output)
            else:
                decoder_output = self.decoder_te(encoder_output)

            return decoder_output
