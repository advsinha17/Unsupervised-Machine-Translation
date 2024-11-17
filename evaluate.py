from src.utils.decodertokens import UNMTDecoderTokens
from src.utils.dataloader import UNMTDataset, data_collate
from transformers import XLMRobertaTokenizerFast
from torch.utils.data import DataLoader
from src.models.models import SEQ2SEQ
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from datasets import load_dataset
from tqdm import tqdm
import json

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
    with open('tgt_vocab.json', 'w') as f:
        json.dump(t, f)

        vocab = tokenizer.get_vocab()

    id_to_token = {v: k for k, v in vocab.items()}

    with open('tokenizer_vocab.json', 'w') as f:
        json.dump(id_to_token, f, ensure_ascii=False, indent=4)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for batch in tqdm(src_dataloader):
            input_ids, attention_mask, _, _, _, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            # print('input_ids: ', input_ids)
            translated_ids = model(lang_list.index(tgt_lang), input_ids, attention_mask)
            # translated_ids = torch.argmax(translated_ids, dim=-1).tolist()
            # print(translated_ids.shape)

            for batch in translated_ids:
                # print("batch: ", batch[0])
                translated_tokens = []
                sent = []
                for token in batch[0]:
                    # print(token)
                    # if tgt_decoder_tokens.id_to_tokenizer.get(tid, tokenizer.pad_token_id) == tokenizer.eos_token_id:
                    #     translated_tokens.append(tokenizer.eos_token_id)
                    #     break
                    translated_tokens.append(tgt_decoder_tokens.id_to_tokenizer.get(token.item(), tokenizer.pad_token_id))
                    # sent.append(tokenizer.decode([tgt_decoder_tokens.id_to_tokenizer.get(token.item(), tokenizer.pad_token_id)]))

                # print('translated tokens: ', translated_tokens)
                translated_text = tokenizer.decode(translated_tokens, skip_special_tokens=True)
                # translated_text = ' '.join(sent)
                print(translated_text)
                translations.append(translated_text)


            # for ids in translated_ids:
            #     translated_tokens = []
            #     for tid in ids:
            #         # if tgt_decoder_tokens.id_to_tokenizer.get(tid, tokenizer.pad_token_id) == tokenizer.eos_token_id:
            #         #     translated_tokens.append(tokenizer.eos_token_id)
            #         #     break
            #         translated_tokens.append(tgt_decoder_tokens.id_to_tokenizer.get(tid, tokenizer.pad_token_id))
            #     print('tokens: ', translated_tokens)
            #     sent = []
            #     for token in translated_tokens:
            #         sent.append(tokenizer.decode([token]))
            #     translated_text = ' '.join(sent)
            #     # translated_text = tokenizer.decode(translated_tokens, skip_special_tokens=True)
            #     translations.append(translated_text)
            #     print('translated: ', translated_text)

    for ref in tgt_text:
        references.append([ref.split()])

    smooth = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, [t.split() for t in translations], weights=(1, 0, 0, 0), smoothing_function=smooth)
    print(f"BLEU score: {bleu_score}")

    return bleu_score


def eval_en_hi_te():
    bleu_scores = {}
    dataset = load_dataset("ai4bharat/samanantar", 'hi', split='train', streaming=True, trust_remote_code=True)
    shuffled_dataset = dataset.shuffle(buffer_size=10000, seed=42)
    subset_dataset = shuffled_dataset.take(1000)
    hi_data = [item['src'] for item in subset_dataset]
    en_data = [item['tgt'] for item in subset_dataset]
    data_size = len(en_data)
    # bleu_scores['en_to_hi'] = evaluate('trained_models/en_hi_te.pth', 'en', en_data, 'hi', hi_data, data_size)
    bleu_scores['hi_to_en'] = evaluate('trained_models/en_hi_te.pth', 'hi', hi_data, 'en', en_data, data_size)

    # dataset = load_dataset("ai4bharat/samanantar", 'te', split='train', streaming=True, trust_remote_code=True)
    # shuffled_dataset = dataset.shuffle(buffer_size=10000, seed=42)
    # subset_dataset = shuffled_dataset.take(1000)
    # te_data = [item['src'] for item in subset_dataset]
    # en_data = [item['en'] for item in subset_dataset]
    # data_size = len(en_data)
    # bleu_scores['en_to_te'] = evaluate('trained_models/en_hi_te.pth', 'en', en_data, 'te', te_data, data_size)
    # bleu_scores['te_to_en'] = evaluate('trained_models/en_hi_te.pth', 'te', te_data, 'en', en_data, data_size)

    # dataset = load_dataset("ai4bharat/IN22-Gen", "tel_Telu-hin_Deva", trust_remote_code=True)
    # te_data = []
    # hi_data = []
    # for item in dataset['gen']:
    #     te_data.append(item['sentence_tel_Telu'])
    #     hi_data.append(item['sentence_hin_Deva'])
    # data_size = len(te_data)
    # bleu_scores['te_to_hi'] = evaluate('trained_models/en_hi_te.pth', 'te', te_data, 'hi', hi_data, data_size)
    # bleu_scores['hi_to_te'] = evaluate('trained_models/en_hi_te.pth', 'hi', hi_data, 'te', te_data, data_size)

    return bleu_scores


if __name__ == '__main__':
    bleu_scores = eval_en_hi_te()
    for k, v in bleu_scores.items():
        print(f"{k}: {v}")









    