"""main func for training"""
import os
import re
import time
import random
import pprint
import datetime
import argparse

from tqdm import tqdm
import ujson
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import DataLoader

import config as cfg
import model as m
import util
import dataset




if __name__ == '__main__':

    cfg.init_model_path()

    parser = argparse.ArgumentParser(description='func for training...')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
    parser.add_argument('--path', type=str, default=None, help='model path')
    parser.add_argument('--root', type=str, default=None, help='root path')

    verbose = False

    args = parser.parse_args()

    gpu_id = args.gpu_id
    gpu_id = int(gpu_id)

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu_id))
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(gpu_id))
    else:
        print('no gpus available')
        device = torch.device('cpu')
    
    
    max_seq_len = 128 # int(args.path.split('t_')[-2].split('_')[-1])
    print('current max_seq_len: {}'.format(max_seq_len))

    if args.path:
        model_paths = [args.path]
        print(model_paths)
    elif args.root:
        model_paths = os.listdir(args.root)
        model_paths = [os.path.join(args.root, path) for path in model_paths]
        print(model_paths)

    print('\n\n') 

    # [START] for model_path in model_paths:
    for idx, model_path in enumerate(model_paths):
        print()
        print('model_path: {}'.format(model_path))
        print(os.listdir(model_path))
        finished = sum([1 for path in os.listdir(model_path) if 'res' in path]) > 0
        if not finished:
            print('not finished yet', model_path)
            continue
        
        for path in os.listdir(model_path):
            if 'res' in path:
                print('CV -> ', path)
        
        models = []

        print('Cur model path:', model_path)
        
        root = cfg.INPUT_BASE
        test_data = pd.read_csv('{}train.csv'.format(root))

        test_data['text'] = test_data.apply(lambda row: str(row.text).strip(), axis=1)

        for i in range(cfg.K_FOLD):
            _model = m.SentimentExtractor(
                model_type=cfg.MODEL_TYPE,
                dropout_rate=cfg.DROPOUT_RATE,
                last_n_layers=cfg.LAST_N_LAYERS,
                device=device)
            _model.to(device)
            _model.load_state_dict(torch.load(f'{model_path}/model_{i}.pt'))
            _model.eval()
            models.append(_model)

        m.init_tokenizer()

        test_dataset = dataset.TweetDataset(
            tweet=test_data.text.values,
            sentiment=test_data.sentiment.values,
            selected_text=test_data.selected_text.values,
            tokenizer=m.tokenizer,
            max_seq_len=max_seq_len,
            model_type=cfg.MODEL_TYPE,
        )

        id_list = []
        answer = []
        sentiments = ['positive', 'negative', 'neutral']
        
        scores = []
        selected = []
        # [START] with torch.no_grad():        
        with torch.no_grad():        
            for idx, d in enumerate(tqdm(test_dataset, desc="test", ncols=80)):

                uniq_id =  test_data.textID.iloc[idx]
                ids = d["ids"]
                token_type_ids = d["token_type_ids"]
                mask = d["mask"]
                sentiment = d["sentiment"]
                orig_selected = d["orig_selected"]
                orig_tweet = d["orig_tweet"]
                targets_start = d["targets_start"]
                targets_end = d["targets_end"]
                offsets = d["offsets"].numpy()

                ids = torch.unsqueeze(ids, dim=0).to(device, dtype=torch.long)
                token_type_ids = torch.unsqueeze(token_type_ids, dim=0).to(device, dtype=torch.long)
                mask = torch.unsqueeze(mask, dim=0).to(device, dtype=torch.long)

                targets_start = targets_start.to(device, dtype=torch.long)
                targets_end = targets_end.to(device, dtype=torch.long)     

                c = [] # sentiment classification
                s = [] # start idx
                e = [] # end idx
                for _model in models:
                    start_logits, end_logits, _ = _model(ids,
                        token_type_ids=token_type_ids,
                        attention_mask=mask)

                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

                    s.append(start_logits)
                    e.append(end_logits)

                s_merged_logits = sum(s)/len(s)
                e_merged_logits = sum(e)/len(e)

                outputs_start = s_merged_logits
                outputs_end = e_merged_logits

                idx_start = np.argmax(outputs_start[0, :])
                idx_end = np.argmax(outputs_end[0, :])

                score, output_sentence = util.calculate_jaccard_score(
                    original_tweet=orig_tweet,
                    target_string=orig_selected,
                    sentiment_val=sentiment,
                    idx_start=idx_start,
                    idx_end=idx_end,
                    offsets=offsets
                )

                scores.append(score)

                if verbose and sentiment != 'neutral' and score < 1.0: 
                    print()
                    print(uniq_id)
                    print(orig_tweet)
                    print('Answ:', orig_selected)
                    print('Pred:', output_sentence, '({}, {})'.format(sentiment, score))

                id_list.append(uniq_id)
                answer.append(output_sentence)

                if idx % 100:
                    df_res = pd.DataFrame.from_dict({
                        'textID': id_list,
                        'selected_text': answer, 
                    })
                    df_res.to_csv('7125res.csv', index=False)

            print('=> avg score:', sum(scores)/len(scores))
            print('------------------------------------------------------')

            df_res = pd.DataFrame.from_dict({
                'textID': id_list,
                'selected_text': answer, 
            })
            df_res.to_csv('7125res.csv', index=False)

        # [END] with torch.no_grad():        
    # [END] for model_path in model_paths:




