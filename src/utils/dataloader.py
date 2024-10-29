from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch 

class UNMTDataset(Dataset):
    def __init__(self, tokenizer: BertTokenizer, lang: str, size: int = 10000):
        super(UNMTDataset, self).__init__()
        assert lang in ['en', 'te', 'hi'], "lang must be one of 'en', 'te', or 'hi'"
        dataset = load_dataset("statmt/cc100", lang = lang, split = "train", streaming = True, trust_remote_code = True)
        self.data = list(dataset.take(size))
        self.size = size
        self.tokenizer = tokenizer
        self.lang = lang

    def __len__(self):
        return self.size
    
    def _tokenize_line(self, line):
        split_line = line.split()
        tokenized_line = self.tokenizer(split_line, is_split_into_words = True)
        word_idx = tokenized_line.word_ids()
        return tokenized_line, word_idx

    def __getitem__(self, idx):
        line = self.data[idx]
        tokenized_line, word_idx = self._tokenize_line(line['text'])
        input_ids = tokenized_line['input_ids']
        attention_mask = tokenized_line['attention_mask']
        return input_ids, attention_mask, word_idx
    

def data_collate(batch, tokenizer):
    input_ids = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
    word_idx = [item[2] for item in batch]
    return input_ids, attention_mask, word_idx
