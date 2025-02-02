from datasets import load_dataset
from transformers import XLMRobertaTokenizerFast
import os
import pickle
from tqdm import tqdm

class UNMTDecoderTokens:
    def __init__(self, data, tokenizer: XLMRobertaTokenizerFast, lang: str, max_examples: int = 100000):
        
        # self.dataset = load_dataset("statmt/cc100", lang = lang, split = 'train', streaming = True, trust_remote_code = True)
        #print(f"Dataset loaded: {self.dataset}")
        self.dataset =  data
        self.tokenizer = tokenizer
        self.lang = lang
        self.token_set = set()
        self.subfolder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decoder_tokens")

        # self.subfolder_name = "./decoder_tokens"
        self.max_examples = max_examples
        self.id_to_tokenizer = {} # maps local id to tokenizer id
        self.tokenizer_to_id = {} # maps tokenizer id to local id 

    def _process_data(self):

        # current_id = 1
        with tqdm(desc=f"Processing train", unit=" examples", ncols=80) as pbar:
            for example in self.dataset:
                if pbar.n >= self.max_examples:
                    break
                sentence = example
                tokenized_line = self.tokenizer(sentence)
                self.token_set.update(tokenized_line['input_ids'])
                pbar.update(1)
        self.token_set = list(sorted(self.token_set))
        for i, token in enumerate(self.token_set):
            self.tokenizer_to_id[token] = i
            self.id_to_tokenizer[i] = token
        self.tokenizer_to_id[self.tokenizer.unk_token_id] = len(self.token_set)
        self.id_to_tokenizer[len(self.token_set)] = self.tokenizer.unk_token_id
        
        
    def create_token_list(self):
        self._process_data()
        if not os.path.exists(self.subfolder_name):
            os.makedirs(self.subfolder_name)
        
        file_path = os.path.join(self.subfolder_name, f"{self.lang}_decoder_tokens_list.pkl")

        with open(file_path, 'wb') as f:
            data = {
                'token_set': self.token_set,
                'id_to_tokenizer': self.id_to_tokenizer,
                'tokenizer_to_id': self.tokenizer_to_id
            }
            pickle.dump(data, f)

        return len(self.token_set)
    
    #this function returns a list, so, self.token_set now is actually a list
    def load_token_list(self):
        file_path = os.path.join(self.subfolder_name, f"{self.lang}_decoder_tokens_list.pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.token_set = list(data['token_set'])
            self.id_to_tokenizer = data['id_to_tokenizer']
            self.tokenizer_to_id = data['tokenizer_to_id']
        
        return self.token_set, self.id_to_tokenizer, self.tokenizer_to_id