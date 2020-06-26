"""main func for training"""
import os
import gc
import re
import sys
import time
import random
import pprint
import shutil
import datetime
import argparse

import ujson
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from torch.utils.data import DataLoader

import config as cfg
import model as m
import engine
import util
import dataset 


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('\n\ncurrent torch\'s random seed {}'.format(torch.cuda.initial_seed()))


def save_code(output_path):
    for name in os.listdir('./'):
        if '.py' in name:
            shutil.copy(name, output_path)
            print(name, 'saved!')


def freeze_transformer(model):
    for n, p in model.named_parameters():
        if 'bert' in n:
            p.requires_grad = False
    print('freeze transformer')


def unfreeze_transformer(model):
   for n, p in model.named_parameters():
       if 'bert' in n:
           p.requires_grad = True
   print('unfreeze all')


def get_dataloader(
    df_train,
    df_valid,
    max_seq_len=128,
    model_type='roberta',
    dataloader_shuffle=False,
    ):

    tokenizer = m.init_tokenizer(model_type) # [!]

    train_dataset = dataset.TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        model_type=model_type,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=dataloader_shuffle,
        num_workers=4,
        drop_last=True
    )

    valid_dataset = dataset.TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        model_type=model_type,
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        drop_last=True
    )

    return train_data_loader, valid_data_loader


def get_single_optim_sched(
    model,
    num_data,
    lr,
    total_epochs,
    batch_size,
    grad_accum_step,
    warmup_frac,
    adam_decay_rate=0.01):

    num_train_steps = int(num_data*total_epochs/(batch_size*grad_accum_step))

    params = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    is_backbone = lambda n: 'bert' in n

    lr_transformer = lr
    lr_head = lr*500

    optimizer_parameters = [
        {
            'params': [p for n, p in params if is_backbone(n) and not any(nd in n for nd in no_decay)],
            'weight_decay': adam_decay_rate, 
            'lr': lr_transformer,
        },
        {
            'params': [p for n, p in params if is_backbone(n) and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': lr_transformer,
        },
        {
            'params': [p for n, p in params if not is_backbone(n)],
            'weight_decay': adam_decay_rate, 
            'lr': lr_head,
        },
    ]

    optimizer = AdamW(optimizer_parameters)

    num_warmup_steps = int(num_train_steps*warmup_frac)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=num_train_steps)

    return optimizer, lr_scheduler


def get_optim_sched_at(
    model,
    epoch_i,
    num_data,
    lr_each_epochs=[1e-5, 1e-5, 5e-6, 3e-6],
    use_sched_each_epochs=[False, False, True, True],
    adam_decay_rate=0.01):

    num_train_steps = int(num_data/args.batch_size)

    params = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    is_backbone = lambda n: 'bert' in n

    optimizer_parameters = [
        {
            'params': [p for n, p in params if is_backbone(n) and not any(nd in n for nd in no_decay)],
            'weight_decay': adam_decay_rate,
            'lr': lr_each_epochs[epoch_i],
        },
        {
            'params': [p for n, p in params if is_backbone(n) and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': lr_each_epochs[epoch_i],
        },
        {
            'params': [p for n, p in params if not is_backbone(n)],
            'weight_decay': adam_decay_rate,
            'lr': lr_each_epochs[epoch_i]*500,
        },
    ]

    optimizer = AdamW(optimizer_parameters)
    
    lr_scheduler = None 
    if use_sched_each_epochs[epoch_i]:
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=0,
                                                       num_training_steps=num_train_steps)

    print('[Epoch {}, lr {}, sche {}]'.format(epoch_i, lr_each_epochs[epoch_i], lr_scheduler))

    return optimizer, lr_scheduler




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='func for training...')
    parser.add_argument('--memo', type=str, default='test', help='memo')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
    parser.add_argument('--seed_list', type=str, default='42', help='seed list as str seperated by comma')
    parser.add_argument('--input', type=str, default='train.csv', help='input data file name')
    parser.add_argument('--input_base', type=str, default='/DATA/image-search/kgg/input/', help='input base path')
    parser.add_argument('--output_base', type=str, default='/DATA/image-search/kgg/output/', help='output base path')
    parser.add_argument('--k_fold', type=int, default=5, help='k-fold num')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--adam_decay_rate', type=float, default=0.01, help='Adam decay rate')
    parser.add_argument('--warmup_frac', type=float, default=0.06, help='warmup fraction for AdamW, not used now')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max seq len')
    parser.add_argument('--model_type', type=str, default='roberta', help='backbone model type')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout for hidden output')
    parser.add_argument('--last_n_layers', type=int, default=2, help='use last_n_layers of hidden output')
    parser.add_argument('--dataloader_shuffle', type=bool, default=True, help='dataloader shuffle')
    parser.add_argument('--use_pseudo_label', type=bool, default=False, help='use pseudo label')
    parser.add_argument('--pseudo_label_path', type=str, default='./dataset/pseudo_selected.csv', help='pseudo label (csv) path')
    parser.add_argument('--es_patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--es_delta', type=float, default=0.001, help='early stopping delta')
    parser.add_argument('--freeze_at_first_epoch', type=bool, default=False, help='freeze transformer at first epoch')
    parser.add_argument('--label_smoothing', type=bool, default=True, help='use truncated label smoothing')
    parser.add_argument('--label_smoothing_ratio', type=float, default=0.2, help='label smoothing ratio')
    parser.add_argument('--use_seq_loss', type=bool, default=True, help='use sequence loss')
    parser.add_argument('--use_dist_loss', type=bool, default=False, help='use sequence loss')
    parser.add_argument('--adv', type=bool, default=False, help='adversarial training')

    # _lr_each_epochs = [1e-5, 1e-5, 4e-6, 2e-6] # squad 8b 0.7125
    _lr_each_epochs = [1e-5, 1e-5, 5e-6, 3e-6] # squad 8b 0.7129
    _use_sched_each_epochs = [False, False, True, True]

    args = parser.parse_args()

    gpu_id = args.gpu_id
    gpu_id = int(gpu_id)

    seeds = [int(seed) for seed in args.seed_list.split(',')]

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu_id))
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(gpu_id))
    else:
        print('no gpus available')
        device = torch.device('cpu')

    kfold_stats = []
    
    train_data = pd.read_csv(os.path.join(args.input_base, args.input))
    train_data.dropna(inplace=True)

    train_data['text'] = train_data.apply(lambda row: str(row.text).strip(), axis=1)
    train_data['selected_text'] = train_data.apply(lambda row: str(row.selected_text).strip(), axis=1)

    # [START] for seed ~ (
    for seed in seeds:

        print("NEW SEED {} -------------------------------------------------------------------".format(seed))

        set_random_seed(seed)

        lr = _lr_each_epochs[0]
        lr_head = lr*500

        output_path = cfg.make_kfold_output_dir(args.output_base, args.memo, args.model_type, seed)

        save_code(output_path)

        skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=seed)

        if args.use_pseudo_label:
            df_pseudo = pd.read_csv(args.pseudo_label_path)
            df_pseudo = df_pseudo.sample(frac=1, random_state=seed).reset_index(drop=True)

        # [START] for k_fold_i ~ (
        for k_fold_i, (train_index, val_index) in enumerate(skf.split(train_data, train_data.sentiment.values)):

            print('\n\n[START] {}-fold'.format(k_fold_i+1))

            df_train = train_data.iloc[train_index].reset_index(drop=True)
            df_valid = train_data.iloc[val_index].reset_index(drop=True)

            if args.use_pseudo_label:
                df_train = pd.concat([df_pseudo, df_train], axis=0, ignore_index=True)
                df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)

            train_data_loader, valid_data_loader = get_dataloader(
                df_train,
                df_valid,
                max_seq_len=args.max_seq_len,
                model_type=args.model_type,
                dataloader_shuffle=args.dataloader_shuffle)

            model = m.SentimentExtractor(
                model_type=args.model_type,
                max_seq_len=args.max_seq_len,
                dropout_rate=args.dropout,
                last_n_layers=args.last_n_layers)
            model.to(device)

            es = util.EarlyStopping(patience=args.es_patience, mode="max", delta=args.es_delta)

            if args.freeze_at_first_epoch:
                freeze_transformer(model)

            best_stat = None

            total_epochs = len(_lr_each_epochs)

            # [START] for epoch_i ~ (
            for epoch_i in range(0, total_epochs):
                print('\n[{}-fold {}-epoch] Training'.format(k_fold_i+1, epoch_i+1))
                print('{} epoch start-------------------------'.format(epoch_i+1))
                print('output path: {}'.format(output_path))

                if args.freeze_at_first_epoch and epoch_i == 1:
                    unfreeze_transformer(model)

                optimizer, lr_scheduler = get_optim_sched_at(
                    model,
                    epoch_i,
                    num_data=len(train_index),
                    lr_each_epochs=_lr_each_epochs,
                    use_sched_each_epochs=_use_sched_each_epochs,
                    adam_decay_rate=args.adam_decay_rate)

                train_stat = engine.do_train_at(k_fold_i,
                                               epoch_i,
                                               train_data_loader,
                                               model,
                                               es,
                                               optimizer,
                                               lr_scheduler,
                                               args.max_seq_len,
                                               device,
                                               adv=args.adv,
                                               use_dist_loss=args.use_dist_loss,
                                               use_seq_loss=args.use_seq_loss,
                                               ce_smoothing=args.label_smoothing,
                                               ce_smoothing_ratio=args.label_smoothing_ratio)

                print('[{}-fold {}-epoch] Validation'.format(k_fold_i+1, epoch_i+1))

                valid_stat = engine.do_validation_at(
                    valid_data_loader,
                    model,
                    args.max_seq_len,
                    device,
                    use_seq_loss=args.use_seq_loss,
                    ce_smoothing=args.label_smoothing,
                    ce_smoothing_ratio=args.label_smoothing_ratio)

                es(valid_stat['jaccard'], model, model_path= '{}/model_{}.pt'.format(output_path, k_fold_i))
                if es.early_stop:
                    print('Early Stopped')
                    break
                elif best_stat is None:
                    best_stat = {
                        'train': train_stat,
                        'valid': valid_stat,
                    }
                    pprint.pprint(best_stat)
                elif best_stat['valid']['jaccard']+es.delta < valid_stat['jaccard']:
                    best_stat = {
                        'train': train_stat,
                        'valid': valid_stat,
                    }
                    pprint.pprint(best_stat)
                else:
                    print('Valid score is not improved! :(')
                    tmp = {
                        'train': train_stat,
                        'valid': valid_stat,
                    }
                    pprint.pprint(tmp)

            # [END] for epoch_i ~ )
            kfold_stats.append(best_stat)
            torch.cuda.empty_cache()
            gc.collect()
        # [END] for k_fold_i ~ )

        scores = [stat['valid']['jaccard'] for stat in kfold_stats]
        avg_score = round(sum(scores)/len(scores), 4)
        with open('{}/res_{}_{}_score_{}.json'.format(output_path, seed, args.model_type, avg_score), 'w') as f:
            f.write(ujson.dumps(kfold_stats))
            print('\n[Avg Val Score] -> {}\n'.format(avg_score))
            kfold_stats = []
            print('TRAIN FIN by {}'.format(seed))
        print('{}-fold result: {}'.format(args.k_fold, kfold_stats))
    # [END] for seed ~ ) 
