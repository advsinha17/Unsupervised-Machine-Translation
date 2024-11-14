import argparse
import os
import json

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

    if num_langs != len(langs):
        raise ValueError(f"num_langs ({num_langs}) does not match the number of languages ({len(langs)})")
    if dataset_sizes and num_langs != len(dataset_sizes):
        raise ValueError(f"num_langs ({num_langs}) does not match the number of dataset sizes ({len(dataset_sizes)})")

    config['dataset_sizes'] = dataset_sizes


def get_parser():
    parser = argparse.ArgumentParser(description = 'Machine Translation System')
    parser.add_argument('-mode', choices = ['train', 'test'], required = True, help = 'train or test mode')

    parser.add_argument('-src', type = str, help = "Source language code (eg. 'en' for English, 'hi' for Hindi)")
    parser.add_argument('-tgt', type = str, help = "Target language code (eg. 'en' for English, 'hi' for Hindi)")


    parser.add_argument('-config', type = str, default = 'config.json', help = "Path to model config file for training")


    parser.add_argument('-load_model', type = str, default = 'model.pth', help = "Path to load trained model")

    

    args = parser.parse_args()

    if args.mode == 'train':
        if args.config is None:
            parser.error("Config file must be specified by -config in train mode")
        config = load_model_config(args.config)
        validate_config(config)

    elif args.mode == 'test':
        if args.src is None or args.target is None:
            parser.error("Source and target languages must be specified by -src and -target in test mode")
    return parser


def run():
    # TODO

    pass



if __name__ == '__main__':
    run()

