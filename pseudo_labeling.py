"""main func for training"""
import os
import re
import sys
import time
import random
import pprint
import datetime
import argparse

import ujson
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
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
    
    max_seq_len = cfg.MAX_SEQ_LEN

    model_path = args.path

    print('\n\n') 

    print()
    print('model_path: {}'.format(model_path))
    print(os.listdir(model_path))
    finished = sum([1 for path in os.listdir(model_path) if 'res' in path]) > 0
    if not finished:
        print('not finished yet', model_path)
        sys.exit()
    
    for path in os.listdir(model_path):
        if 'res' in path:
            print('CV -> ', path)
    
    models = []

    print('Cur model path:', model_path)
    
    root = cfg.INPUT_BASE

    # [TODO] pseudo_senti.csv 로드
    ###################################################################################
    df_pseudo = pd.read_csv(os.path.join('./dataset', 'sent140.csv'))

    df_pseudo.loc[:,'selected_text'] = df_pseudo.text.values

    test_data = df_pseudo.sample(frac=1, random_state=42).reset_index(drop=True)

    ids = []
    for idx, text in enumerate(test_data.text):
        row = test_data.iloc[idx]
        if not util.is_english(row.text):
            ids.append(idx)
    test_data = test_data.drop(ids) 

    print(test_data)
    ###################################################################################

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
    
    selecteds = []
    # [START] with torch.no_grad():        
    with torch.no_grad():        
        for idx, d in enumerate(tqdm(test_dataset, desc="Pseudo", ncols=80)):

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

            # 확률 차이 고려해서 추출?
            #_mask = mask.cpu().detach().numpy()[0]
            #print(np.where(_mask == 0))

            idx_start = np.argmax(outputs_start[0, :])
            idx_end = np.argmax(outputs_end[0, :])

            _, output_sentence = util.calculate_jaccard_score(
                original_tweet=orig_tweet,
                target_string=orig_selected,
                sentiment_val=sentiment,
                idx_start=idx_start,
                idx_end=idx_end,
                offsets=offsets
            )

            selecteds.append(output_sentence)
            
            with open('./pseudo_tmp.out', 'a') as f:
                if idx % 50 == 0:
                    f.write("[{}] {}\n=> {}\n\n".format(sentiment, orig_tweet, output_sentence))

    # [END] with torch.no_grad():        

    test_data.loc[:, 'selected_text'] = selecteds 
    test_data.to_csv('./dataset/pseudo_selected.csv', index=False)




