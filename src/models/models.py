import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel
from torch.nn.utils.rnn import pad_sequence
from src.utils.decodertokens import UNMTDecoderTokens

class UNMTEncoder(nn.Module):
    def __init__(self, pretrained_type: str = 'xlm-roberta-base'):
        '''
            pretrained_type: str, type of xlmr model to use
        '''
        super(UNMTEncoder, self).__init__()
        self.xlmr = XLMRobertaModel.from_pretrained(pretrained_type)
        
        for param in self.xlmr.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.xlmr(input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state
        return x  # returns (batch, seqlen, 768)

class LSTM_ATTN_Decoder(nn.Module):
    def __init__(self, lang:str , tokenizer: XLMRobertaTokenizerFast, embedding_layer: torch.nn.modules.sparse.Embedding,
                 max_output_length:int = 50 , dropout_rate:float = 0.3 ,num_layers:int = 3,
                 embedding_dim: int = 768, hidden_dim:int = 1024):
        
        super(LSTM_ATTN_Decoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding_layer = embedding_layer

        self.lang = lang
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.tokenizer = tokenizer

        self.start_token_id = self.tokenizer.cls_token_id
        self.end_token_id = self.tokenizer.sep_token_id

        self.decoderTokens = UNMTDecoderTokens(None, tokenizer = self.tokenizer, lang = lang)
        self.decoderTokens.load_token_list()
        self.decoderTokens.token_set = torch.tensor(self.decoderTokens.token_set).to(self.device) #1D tensor of valid tokens
        
        self.beam_width = 3
        self.beam_length_norm = 0.7
        self.epsilon = 1e-7

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

        outputs = []

        if self.training:

            #initializing LSTM hidden and cell states
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim ).to(self.device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim ).to(self.device)

            
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
            
            outputs = torch.stack(outputs, dim = 1)

            return outputs

        else:
            
            for batch_idx in range(batch_size):
                #initializing LSTM hidden and cell states
                h = torch.zeros((self.num_layers, 1, self.hidden_dim ), requires_grad=False).to(self.device) # as we're iterating through a batch sample by sample
                c = torch.zeros((self.num_layers, 1, self.hidden_dim ), requires_grad=False).to(self.device)

                beams = [(torch.zeros((1,0), dtype = torch.long, requires_grad=False).to(self.device), torch.zeros((1),requires_grad=False).to(self.device), h, c)]
                completed_beams = []
                # (seq, score, hidden, cell) # seq is [1,current_seq_len] # score is [1]
                for t in range(self.max_output_length):
                    new_beams = []
                    #the 1 in the first dim of everything is for the batch_dim
                    for(seq, score, h, c) in beams:
                        attn_scores = []
                        for i in range(seq_len1):
                            #print(f"x[{batch_idx}, {i}, :] shape is {x[batch_idx, i, :].shape}, h[-1] shape is {h[-1].shape}")
                            attn_input = torch.cat((x[batch_idx, i, :].unsqueeze(0), h[-1]), dim = 1)  # h[-1] is the hidden state from last layer
                            #attn_input has the shape [1, embedding_dim + hidden_dim]

                            attn_scores.append(self.Attention_MLP(attn_input)) # seq_len1 no of elements appended to the list; of shape [1, 1]

                        attn_scores_tensor = torch.stack(attn_scores, dim = 1) #[1, seq_len1, 1]
                        attn_scores_tensor = self.softmax_layer(attn_scores_tensor)

                        context_vector = torch.sum(attn_scores_tensor * x[batch_idx, :, :].unsqueeze(0), dim = 1) #[1, embedding_dim]

                        if t==0:
                            current_intput_token =torch.tensor([self.start_token_id]*1).to(self.device) #[1]
                            current_input = self.embedding_layer(current_intput_token).to(self.device) #[1, embedding_dim]
                        else:
                            #takes the last tokens in seq for the whole batch
                            prev_pred_tokens = self.decoderTokens.token_set[seq[:,-1]] #[1 embedding_dim]
                            #if the prev_pred_token was the end token then we dont want to use it as input
                            # we then add it to the completed beams
                            if prev_pred_tokens[0].item() == self.decoderTokens.tokenizer_to_id[self.tokenizer.eos_token_id]:
                                completed_beams.append((seq, score, h, c))
                                continue
                            current_input = self.embedding_layer(prev_pred_tokens).to(self.device)  #[batch_size, embedding_dim]
                        #print(f"current_input shape is {current_input.shape}, context_vector shape is {context_vector.shape}")
                        lstm_input = torch.cat((current_input, context_vector), dim = 1).unsqueeze(1) #[1, 1 ,2*embedding_dim]
                        lstm_output, (h,c) = self.LSTM(lstm_input, (h,c))
                        pred_prob = self.fc_final(lstm_output.squeeze(1)) #[1, no_output_tokens]
                        pred_prob_softmax = self.softmax_layer(pred_prob) #[1, no_output_tokens]
                        #but pred_prob_softmax is actually the logSoftmax
                        #for beam search we require normal softmax.
                        pred_prob_softmax = torch.exp(pred_prob_softmax)

                        top_k_probs, top_k_indices = torch.topk(pred_prob_softmax, self.beam_width, dim=-1) # [batch, beam_width]
                        # print("top_k_probs.shape is ", top_k_probs.shape)
                        # print("top_k_indices.shape is ", top_k_indices.shape)
                        for i in range(self.beam_width):
                            new_seq = torch.cat([seq, top_k_indices[:,i].unsqueeze(1)],dim = 1)
                            new_length = new_seq.size(1)
                            penalty = ((5+new_length)/6)**self.beam_length_norm
                            new_score = score + torch.log(top_k_probs[:,i]+self.epsilon)/penalty
                            assert not torch.isnan(new_score).any(), "NaN values found in score"
                            #new_score = score + torch.log(top_k_probs[:,i])
                            new_beams.append((new_seq, new_score, h, c))

                    # now we have to select the top beam_width beams
                    top_beams_per_sample = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.beam_width]
                    beams = top_beams_per_sample
                
                completed_beams.extend(beams)
                best_sequence = max(completed_beams, key=lambda x: x[1])
                outputs.append(best_sequence[0].squeeze(0))
            #now we have a list of tensors of shape [1, sample_seq_len] for each sample in the batch
            # we need to pad them all to the same length and stack them
            # print([tensor.shape for tensor in outputs])]
            padded_output = pad_sequence(outputs, batch_first=True, padding_value=-1)
            return padded_output

   
class SEQ2SEQ(nn.Module):
    def __init__(self, tokenizer: XLMRobertaTokenizerFast, list_of_target_languages:list, pretrained_type:str = 'xlm-roberta-base',max_output_length:int = 50,
                 decoder_hidden_dim:int = 768):
        super(SEQ2SEQ, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = UNMTEncoder(pretrained_type = pretrained_type)
        self.embedding_layer = self.encoder.xlmr.embeddings.word_embeddings.to(self.device)
        self.list_of_target_languages = list_of_target_languages
        self.decoders = nn.ModuleList()

        for idx, lang in enumerate(list_of_target_languages):
            self.decoders.append(LSTM_ATTN_Decoder(lang = lang, tokenizer = tokenizer, 
                                     embedding_layer = self.embedding_layer,max_output_length = max_output_length,
                                     hidden_dim = decoder_hidden_dim))

    def forward(self, language:int ,
                input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                noisy_input_ids: torch.Tensor = None, noisy_attention_mask = None, 
                g_truth: bool = False, noisy_input: bool = False):
        #for example: language is 0 for en, 1 for hi and 2 for te and so on
        if self.training:
            
            #so the noisy is actually the input to the encoder in this case.
            #but we dont want noise for backtranslation so we have a flag for that
            if(noisy_input):
                encoder_output = self.encoder(noisy_input_ids, attention_mask = noisy_attention_mask)
            else:
                encoder_output = self.encoder(input_ids, attention_mask = attention_mask)
            
            decoder_output = self.decoders[language](encoder_output, target_seq = input_ids, g_truth = g_truth)
            return decoder_output
        
        else:
            encoder_output = self.encoder(input_ids, attention_mask = attention_mask)
            decoder_output = self.decoders[language](encoder_output)
            return decoder_output
