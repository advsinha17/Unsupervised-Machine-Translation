import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import random

class UNMTDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizerFast, lang: str, 
                 size: int = 10000, shuffle_k: int = 3, p_drop: float = 0.1):
        super(UNMTDataset, self).__init__()
        assert lang in ['en', 'te', 'hi'], "lang must be one of 'en', 'te', or 'hi'"
        dataset = load_dataset("statmt/cc100", lang = lang, split = "train", streaming = True, trust_remote_code = True)
        self.data = list(dataset.take(size))
        self.size = size
        self.tokenizer = tokenizer
        self.lang = lang

        self.shuffle_k = shuffle_k #how much to shuffle the tokens
        self.p_drop = p_drop #probability of dropping a token

    def __len__(self):
        return self.size
    
    def _tokenize_line(self, line):
        split_line = line.split()
        tokenized_line = self.tokenizer(split_line, is_split_into_words = True, return_tensors = 'pt')
        word_idx = tokenized_line.word_ids()
        return tokenized_line, word_idx
    
    def _add_noise(self, tokenized_line, word_idx):
        input_ids = tokenized_line['input_ids'][0].tolist()
        grouped_tokens = [[]]

        length = len(word_idx)
        for idx, word_id in enumerate(word_idx):
            grouped_tokens[-1].append(input_ids[idx])
            if (idx< length - 1) and (word_idx[idx]!= word_idx[idx + 1]):
                grouped_tokens.append([])
        
        # to remove the start and end tokens as well as any padding if present
        first_element = grouped_tokens[0]
        last_element = grouped_tokens[-1]

        del grouped_tokens[0]
        del grouped_tokens[-1]

        
        shuffler_list = [i + random.uniform(0, self.shuffle_k) for i in range(len(grouped_tokens))]
        
        paired = list(zip(grouped_tokens, shuffler_list))
        paired.sort(key = lambda x: x[1])

        grouped_tokens,_ = zip(*paired)
        grouped_tokens = list(grouped_tokens)
        #now grouped_tokens is shuffled
        #now, we have to drop some of these tokens with probability p_drop

        for idx in range(len(grouped_tokens)):
            if random.random() < self.p_drop:
                grouped_tokens[idx] = []
        
        grouped_tokens = [first_element] + grouped_tokens + [last_element]
        
        noisy_tokenized_line = []

        for group in grouped_tokens :
            if group:
                noisy_tokenized_line.extend(group)

        noisy_tokenized_line = torch.tensor(noisy_tokenized_line)
        return noisy_tokenized_line # returned as a tensor
        

    def __getitem__(self, idx):
        line = self.data[idx]
        tokenized_line, word_idx = self._tokenize_line(line['text'])

        input_ids = tokenized_line['input_ids'].squeeze(0) #squeezing to remove the first dimension. That will be added by the dataloader.
        attention_mask = tokenized_line['attention_mask'].squeeze(0)
        noisy_input_ids = self._add_noise(tokenized_line, word_idx)
        noisy_attention_mask = torch.ones_like(noisy_input_ids)

        return input_ids, attention_mask, noisy_input_ids, noisy_attention_mask, line['text'], word_idx

    

def data_collate(batch, tokenizer):
    input_ids = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
    noisy_input_ids = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    noisy_attention_mask = torch.nn.utils.rnn.pad_sequence([item[3] for item in batch], batch_first=True, padding_value=0)
    orig_sentence = [item[4] for item in batch]
    word_idx = [item[5] for item in batch]

    return input_ids, attention_mask, noisy_input_ids, noisy_attention_mask,orig_sentence, word_idx

