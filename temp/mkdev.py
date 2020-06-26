"""main func for training"""
import os
import re
import sys
import time
import random
import pprint
import datetime

import ujson

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

import config as cfg


if __name__ == '__main__':
    
    seed = 42
    root = cfg.INPUT_BASE

    train_data = pd.read_csv('{}train.csv'.format(cfg.INPUT_BASE))
    # train_data = pd.read_csv('/DATA/image-search/kgg/input/preproc/train_pre_v4.csv')
    train_data.dropna(inplace=True)
    train_data['text'] = train_data.apply(lambda row: str(row.text).strip(), axis=1)
    train_data['selected_text'] = train_data.apply(lambda row: str(row.selected_text).strip(), axis=1)

    train_data['len'] = train_data.apply(lambda row: len(row.selected_text)//14, axis=1) # 10개 정도의 범위로... (max=141)
    train_data['ratio'] = train_data.apply(lambda row: int(round(len(row.selected_text)/len(row.text), 1)*10), axis=1)
    train_data['_class'] = train_data.apply(lambda row: "{}_{}_{}".format(row.sentiment, row.len, row.ratio), axis=1)

    #####################################
    random.seed(seed)
    np.random.seed(seed)
    #####################################

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    for k_fold_i, (train_index, val_index) in enumerate(skf.split(train_data, train_data._class.values)):

        df_train = train_data.iloc[train_index].reset_index(drop=True)
        df_dev = train_data.iloc[val_index].reset_index(drop=True)
        
        df_train.to_csv('{}train{}.csv'.format(root, seed), index=False)
        df_dev.to_csv('{}dev{}.csv'.format(root, seed), index=False)
