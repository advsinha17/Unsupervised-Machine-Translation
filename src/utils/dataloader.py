from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer

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
        tokenized_line = self.tokenizer(split_line, is_split_into_words = True, return_tensors = 'pt')
        word_idx = tokenized_line.word_ids()
        return tokenized_line, word_idx

    def __getitem__(self, idx):
        line = self.data[idx]
        tokenized_line, word_idx = self._tokenize_line(line['text'])
        return tokenized_line, word_idx