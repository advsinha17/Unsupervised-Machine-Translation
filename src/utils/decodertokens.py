from datasets import load_dataset
from transformers import BertTokenizerFast
import os
import pickle
from tqdm import tqdm

class UNMTDecoderTokens():
    def __init__(self, tokenizer: BertTokenizerFast, lang: str):
        assert lang in ['en', 'te', 'hi'], "lang must be one of 'en', 'te', or 'hi'"
        
        self.dataset = load_dataset("statmt/cc100", lang = lang, split = 'train', streaming = True, trust_remote_code = True)
        print(f"Dataset loaded: {self.dataset}")
        
        self.tokenizer = tokenizer
        self.lang = lang
        self.token_set = set()
        self.subfolder_name = "decoder_tokens"
        self.max_examples = 1000000
        self.id_to_tokenizer = {} # maps local id to tokenizer id
        self.tokenizer_to_id = {} # maps tokenizer id to local id 

    def _process_data(self):
        print(type(self.dataset))
        current_id = 0
        with tqdm(desc=f"Processing train", unit=" examples", ncols=80) as pbar:
            for example in self.dataset:
                if pbar.n >= self.max_examples:
                    break
                sentence = example['text']
                split_sentence = sentence.split()
                tokenized_line = self.tokenizer(split_sentence, is_split_into_words=True)
                self.token_set.update(tokenized_line['input_ids'])
                for input_id in tokenized_line['input_ids']:
                    if input_id not in self.tokenizer_to_id:
                        self.tokenizer_to_id[input_id] = current_id
                        self.id_to_tokenizer[current_id] = input_id
                        current_id += 1
                pbar.update(1)
        
        # for example in tqdm(self.dataset, desc=f"Processing train", unit="examples", ncols=80):
        #     sentence = example['text']
        #     split_sentence = sentence.split()
        #     tokenized_line = self.tokenizer(split_sentence, is_split_into_words = True)
        #     self.token_set.update(tokenized_line['input_ids'])

    def create_token_list(self):
        self._process_data()
        if not os.path.exists(self.subfolder_name):
            os.makedirs(self.subfolder_name)
        
        file_path = os.path.join(self.subfolder_name, f"{self.lang}_decoder_tokens_list.pkl")

        with open(file_path, 'wb') as f:
            pickle.dump(sorted(self.token_set), f)

        return len(self.token_set)
    
    #this function returns a list
    def load_token_list(self):
        file_path = os.path.join(self.subfolder_name, f"{self.lang}_decoder_tokens_list.pkl")
        with open(file_path, 'rb') as f:
            self.token_set = pickle.load(f)
        
        return self.token_set