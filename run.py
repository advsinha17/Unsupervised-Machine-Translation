import argparse
import os
import json
from datasets import load_dataset
from src.utils.decodertokens import UNMTDecoderTokens
from src.utils.dataloader import UNMTDataset, data_collate
from transformers import XLMRobertaTokenizerFast
from torch.utils.data import DataLoader
import torch
from src.models.models import SEQ2SEQ
from src.trainer import Trainer
from evaluate import evaluate
from tqdm import tqdm

def load_model_config(user_config_file, default_config_file='config.json'):
    with open(default_config_file, 'r') as f:
        default_config = json.load(f)
    
    user_config = {}
    if user_config_file and os.path.exists(user_config_file):
        with open(user_config_file, 'r') as f:
            user_config = json.load(f)
    else:
        print(f"Warning: User config file '{user_config_file}' not found. Using default config values.")
    
    config = {**default_config, **user_config}
    return config

def validate_config(config):
    if 'num_langs' not in config or 'langs' not in config:
        raise ValueError("Config file must contain 'num_langs' and 'langs'")
    
    num_langs = config['num_langs']
    langs = config['langs']
    dataset_sizes = config.get('dataset_sizes', [10000] * num_langs)
    decoder_tokens_sizes = config.get('decoder_tokens_size', [100000] * num_langs)

    if num_langs != len(langs):
        raise ValueError(f"num_langs ({num_langs}) does not match the number of languages ({len(langs)})")
    if dataset_sizes and num_langs != len(dataset_sizes):
        raise ValueError(f"num_langs ({num_langs}) does not match the number of dataset sizes ({len(dataset_sizes)})")

    config['dataset_sizes'] = dataset_sizes
    config['decoder_tokens_size'] = decoder_tokens_sizes

def load_lang_data(lang: str, lang_path: str, dataset_size: int, tokens_size: int, decoder_tokens_exist: bool):
    if lang_path:
        if not os.path.exists(lang_path):
            raise FileNotFoundError(f"Language dataset not found at '{lang_path}'")
        with open(lang_path, 'r') as file:
            data = file.read().split('\n')
        return data[:dataset_size], data[:tokens_size]
    else:
        try:
            dataset = load_dataset("statmt/cc100", lang = lang, split = "train", streaming = True, trust_remote_code = True)
            dl_texts = dataset.take(dataset_size)
            if not decoder_tokens_exist:
                dt_texts = dataset.take(tokens_size)
            dl_data = []
            dt_data = []
            for item in dl_texts:
                dl_data.append(item['text'])
            if not decoder_tokens_exist:
                for item in dt_texts:
                    dt_data.append(item['text'])
                return dl_data, dt_data
            else:
                return dl_data, None
            
        except Exception as e:
            raise ValueError(f"Could not load dataset for language '{lang}'. Please ensure it is a valid language from https://huggingface.co/datasets/statmt/cc100.")
        
def get_model_list():
    if not os.path.isdir('trained_models/'):
        os.makedirs('trained_models/')
    return os.listdir('trained_models/')


def check_model_exists(src, tgt, models):
    for model in models:
        model_langs = model.split('.')[0]
        model_langs = model_langs.split('_')
        if src in model_langs and tgt in model_langs:
            return model
    return None

def get_parser():
    parser = argparse.ArgumentParser(description = 'Machine Translation System')
    parser.add_argument('-mode', choices = ['train', 'translate', 'evaluate'], required = True, help = 'train, translate or evaluate model')


    parser.add_argument('-config', type = str, default = 'config.json', help = "Path to model config file for training")
    parser.add_argument('-language_paths', nargs='*', help="""Paths to language datasets, one per language in the order language names are specified in the config file. If not provided for a language, the cc100 dataset for the language will be used if available.
                        If number of paths exceeds number of languages, the extra paths will be ignored. If a file is provided for a language, each training sample should be on a new line.""")

    models = get_model_list()
    parser.add_argument('-src', type = str, help = f"""Source language code. Both source and target language codes must have a model trained on them. All trained models: {models}. 
                        A language code is the text before each underscore in the model name (eg. a model named en_hi_te can translate between any 2 of en (English), hi (Hindi) and te (Telugu)). .Must be provided in test and evaluate modes""")
    parser.add_argument('-tgt', type = str, help = f"""Target language code. Both source and target language codes must have a model trained on them. All trained models: {models}. 
                        A language code is the text before each underscore in the model name (eg. a model named en_hi_te can translate between any 2 of en (English), hi (Hindi) and te (Telugu)). Must be provided in test and evaluate modes""")
    parser.add_argument('-load_model', type = str, help = "Path to load trained model")
    parser.add_argument('-text', type = str, help = 'Text to be translated.')
    parser.add_argument('-text_file', type = str, help = 'Path to file containing text to be translated.')
    parser.add_argument('-save_file_path', type = str, default = 'Path to save the translated text.')

    parser.add_argument('-src_text_path', type = str, default = 'data/en.txt', help = 'Path to source language text file for evaluation. Each testing sample should be on a new line.')
    parser.add_argument('-tgt_text_path', type = str, default = 'data/hi.txt', help = 'Path to target language text file for evaluation. Each testing sample should be on a new line.')

    
    return parser

def run():
    parser = get_parser()
    args = parser.parse_args()

    if args.mode == 'train':
        if args.config is None:
            parser.error("Config file must be specified by -config in train mode")
        config = load_model_config(args.config)
        validate_config(config)
        dataloaders = []
        tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
        for i, lang in enumerate(config['langs']):
            if args.language_paths: 
                lang_path = args.language_paths[i] if i < len(args.language_paths) else None
            else:
                lang_path = None
            dec_tokens_exist = False
            if os.path.exists(f'src/utils/decoder_tokens/{lang}_decoder_tokens_list.pkl'):
                dec_tokens_exist = True
            dl_data, dt_data = load_lang_data(lang, lang_path, config['dataset_sizes'][i], config['decoder_tokens_size'][i], dec_tokens_exist)
            if not dec_tokens_exist:
                decoder_tokens = UNMTDecoderTokens(dt_data, tokenizer, lang, config['decoder_tokens_size'][i])
                _ = decoder_tokens.create_token_list()
            dataset = UNMTDataset(dl_data, tokenizer, lang, size = config['dataset_sizes'][i])
            dataloader = DataLoader(dataset, batch_size = config['batch_size'], collate_fn = lambda x: data_collate(x, tokenizer))
            dataloaders.append(dataloader)
        trainer = Trainer(dataloaders, tokenizer, lr = config['learning_rate'])
        model = SEQ2SEQ(tokenizer, config['langs'])
        trainer.train(model, config['epochs'])
        print("Model trained successfully.")


    elif args.mode == 'translate':
        if args.src is None or args.tgt is None:
            parser.error("Source and target languages must be specified by -src and -tgt in test mode")
        
        if not args.text and not args.text_file:
            parser.error("Either -text or -text_file must be provided in test mode")
        
        if args.text:
            print(f"Translating text: {args.text}")
            text_to_translate = args.text
        else:
            if not os.path.exists(args.text_file):
                raise parser.error('One of -text or -text_file must be specified')
            with open(args.text_file, 'r') as file:
                text_to_translate = file.read()
            print(f"Translating text from file: {args.text_file}")
        if not args.load_model:
            models = get_model_list()
            model = check_model_exists(args.src, args.tgt, models)
        else:
            model = args.load_model
            model = model.split('/')[-1]
            langs = model.split('.').split('_')
            if args.src not in langs or args.tgt not in langs:
                raise ValueError(f"Model at '{args.load_model}' does not support translation between {args.src} and {args.tgt}")
        if not model:
            raise ValueError(f"No model found in 'trained_models/' directory that supports translation between {args.src} and {args.tgt}")
        tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
        lang_list = model.split('.')[0].split('_')
        print(f"Using model: {model}")
        model = SEQ2SEQ(tokenizer, lang_list) 
        state_dict = torch.load('trained_models/' + model)
        model.load_state_dict(state_dict)
        model.eval()
        tokenized_text = tokenizer(text_to_translate, return_tensor = 'pt')
        input_ids = tokenized_text['input_ids'][0].tolist()
        attention_mask = tokenized_text['attention_mask'][0].tolist()
        translated_ids = model(lang_list.index(args.tgt), input_ids, attention_mask)
        # translated_ids = torch.argmax(translated_ids, dim=-1).tolist()
        decoder_tokens = UNMTDecoderTokens(None, tokenizer, args.tgt)
        decoder_tokens.load_token_list()
        translated_ids = [decoder_tokens.id_to_tokenizer.get(tid, tokenizer.pad_token_id) for tid in translated_ids]
        translated_text = tokenizer.decode(translated_ids, skip_special_tokens=True)
        if not args.save_file_path:
            print(f"Translated text: {translated_text}")
        else:
            with open(args.save_file_path, 'w') as file:
                file.write(translated_text)
            print(f"Translated text saved to: {args.save_file_path}")

    elif args.mode == 'evaluate':
        if args.src is None or args.tgt is None:
            parser.error("Source and target languages must be specified by -src and -tgt in evaluate mode")

        if args.src_text_path is None or args.tgt_text_path is None:
            parser.error("Source and target text files must be specified by -src_text_path and -tgt_text_path in evaluate mode")
        
        if not os.path.exists(args.src_text_path):
            raise FileNotFoundError(f"Source text file not found at '{args.src_text_path}'")
        if not os.path.exists(args.tgt_text_path):
            raise FileNotFoundError(f"Target text file not found at '{args.tgt_text_path}'")
        if not args.load_model:
            models = get_model_list()
            model = check_model_exists(args.src, args.tgt, models)
        else:
            model = args.load_model
            model = model.split('/')[-1]
            langs = model.split('.').split('_')
            if args.src not in langs or args.tgt not in langs:
                raise ValueError(f"Model at '{args.load_model}' does not support translation between {args.src} and {args.tgt}")
        if not model:
            raise ValueError(f"No model found in 'trained_models/' directory that supports translation between {args.src} and {args.tgt}")
        
        with open(args.src_text_path, 'r') as f:
            src_data = f.read().split('\n')
        with open(args.tgt_text_path, 'r') as f:
            tgt_data = f.read().split('\n')
        assert len(src_data) == len(tgt_data) and len(src_data) > 0, "Source and target text files must have the same number of lines and at least 1 line"
        data_size = len(src_data)
        print(f"Bleu score: {evaluate(model, args.src, src_data, args.tgt, tgt_data, data_size)}")


if __name__ == '__main__':
    if not os.path.exists('trained_models/'):
        os.makedirs('trained_models/')
    run()

