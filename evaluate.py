from src.utils.decodertokens import UNMTDecoderTokens
from src.utils.dataloader import UNMTDataset, data_collate
from transformers import XLMRobertaTokenizerFast
from torch.utils.data import DataLoader
from src.models.models import SEQ2SEQ
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from datasets import load_dataset
from tqdm import tqdm

def evaluate(model_path: str, src_lang: str, src_data: list, tgt_lang: str, tgt_text: list, data_size: int):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
    src_decoder_tokens = UNMTDecoderTokens(None, tokenizer, src_lang)
    tgt_decoder_tokens = UNMTDecoderTokens(None, tokenizer, tgt_lang)
    src_decoder_tokens.load_token_list()
    _, t, _ = tgt_decoder_tokens.load_token_list()
    src_dataset = UNMTDataset(src_data, tokenizer, src_lang, size = data_size)
    src_dataloader = DataLoader(src_dataset, batch_size = 8, collate_fn = lambda x: data_collate(x, tokenizer))
    lang_list = model_path.split('/')[1].split('.')[0].split('_')
    print(lang_list)
    model = SEQ2SEQ(tokenizer, lang_list)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    translations = []
    references = []


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for batch in tqdm(src_dataloader):
            input_ids, attention_mask, _, _, _, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            translated_ids = model(lang_list.index(tgt_lang), input_ids, attention_mask)
    
            
            for batch in translated_ids:
                translated_tokens = []
                for token in batch:
                    if token == 1:
                        break
                    if token == -1:
                        translated_tokens.append(tokenizer.pad_token_id)
                        continue
                    translated_tokens.append(tgt_decoder_tokens.id_to_tokenizer.get(token.item(), tokenizer.pad_token_id))
                translated_text = tokenizer.decode(translated_tokens, skip_special_tokens=True)
                print(translated_text)
                translations.append(translated_text)



    for ref in tgt_text:
        references.append([ref.split()])

    smooth = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, [t.split() for t in translations], weights=(1, 0, 0, 0), smoothing_function=smooth)
    print(f"BLEU score: {bleu_score}")

    return bleu_score


def eval_en_hi_te():
    bleu_scores = {}

    with open('data/en_hi.txt', 'r') as f:
        en_data = f.read().split('\n')
    with open('data/hi_en.txt', 'r') as f:
        hi_data = f.read().split('\n')
    bleu_scores['hi_to_en'] = evaluate('trained_models/en_hi_te.pth', 'hi', hi_data[100:600], 'en', en_data[100:600], 500)
    bleu_scores['en_to_hi'] = evaluate('trained_models/en_hi_te.pth', 'en', en_data[100:600], 'hi', hi_data[100:600], 500)

    with open('data/en_te.txt', 'r') as f:
        en_data = f.read().split('\n')
    with open('data/te_en.txt', 'r') as f:
        te_data = f.read().split('\n')
    bleu_scores['en_to_te'] = evaluate('trained_models/en_hi_te.pth', 'en', en_data[:500], 'te', te_data[:500], 500)
    bleu_scores['te_to_en'] = evaluate('trained_models/en_hi_te.pth', 'te', te_data[:500], 'en', en_data[:500], 500)

    dataset = load_dataset("ai4bharat/IN22-Gen", "tel_Telu-hin_Deva", trust_remote_code=True)
    te_data = []
    hi_data = []
    for item in dataset['gen']:
        te_data.append(item['sentence_tel_Telu'])
        hi_data.append(item['sentence_hin_Deva'])
    bleu_scores['te_to_hi'] = evaluate('trained_models/en_hi_te.pth', 'te', te_data[:500], 'hi', hi_data[:500], 500)
    bleu_scores['hi_to_te'] = evaluate('trained_models/en_hi_te.pth', 'hi', hi_data[:500], 'te', te_data[:500], 500)

    return bleu_scores


if __name__ == '__main__':
    bleu_scores = eval_en_hi_te()
    for k, v in bleu_scores.items():
        print(f"{k}: {v}")









    