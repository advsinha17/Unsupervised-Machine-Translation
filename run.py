import argparse
import os
import json

def load_model_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


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

    elif args.mode == 'test':
        if args.src is None or args.target is None:
            parser.error("Source and target languages must be specified by -src and -target in test mode")
    return parser


def run():
    # TODO

    pass



if __name__ == '__main__':
    run()

