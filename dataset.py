"""util for preproc & training"""
import datetime
import collections

import numpy as np
import torch

import util
import config as cfg


SENTIMENTS = ['positive', 'negative', 'neutral']
SENT_TO_IDX = { sent: idx for idx, sent in enumerate(SENTIMENTS)}

_PATTERNS_DICT = collections.OrderedDict([
    ('ï¿½', "'"),
    ('ï', "'"),
    ('Â', '.'),
])

def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    tweet =  " ".join(str(tweet).split())
    selected_text = " ".join(str(selected_text).split())

    tweet, _, _ = util.text_change_pre(tweet, _PATTERNS_DICT)
    selected_text, _, _ = util.text_change_pre(selected_text, _PATTERNS_DICT)

    len_st = len(selected_text)
    idx0 = None
    idx1 = None
    for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
        if tweet[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
    
    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids[1:-1]
    tweet_offsets = tok_tweet.offsets[1:-1]

    input_ids_orig = input_ids_orig[:max_len-4]
    tweet_offsets = tweet_offsets[:max_len-4]

    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)

    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    get_token_id = lambda word: tokenizer.encode(word).ids[0]
    sentiment_id = {
        'positive': get_token_id('positive'), # 3893,
        'negative': get_token_id('negative'), # 4997,
        'neutral': get_token_id('neutral'), # 8699
        'sentiment': get_token_id('sentiment')
    }
    
    input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig + [102]
    token_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]
    targets_start += 3
    targets_end += 3

    targets_seq = [0]*targets_start + [1]*(targets_end-targets_start+1) # [!]
    targets_seq += [0]*(len(input_ids) - len(targets_seq))

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

        targets_seq = targets_seq + [0]*padding_length # [!]
    
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'targets_seq': targets_seq, # [!]
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets
    }


def process_data_roberta(tweet, selected_text, sentiment, tokenizer, max_len):
    tweet = " " + " ".join(str(tweet).split())
    selected_text = " " + " ".join(str(selected_text).split())

    tweet, _, _ = util.text_change_pre(tweet, _PATTERNS_DICT)
    selected_text, _, _ = util.text_change_pre(selected_text, _PATTERNS_DICT)

    len_st = len(selected_text) - 1 # space
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1 # inclusive
            break

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1 # char level check

    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_offsets = tok_tweet.offsets

    input_ids_orig = input_ids_orig[:max_len-5] # 5 == specital tokens cnt
    tweet_offsets = tweet_offsets[:max_len-5]

    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)

    targets_start = target_idx[0]
    targets_end = target_idx[-1] # token level idx

    get_token_id = lambda word: tokenizer.encode(word).ids[0]
    sentiment_id = {
        'positive': get_token_id('positive'), # 1313,
        'negative': get_token_id('negative'), # 2430,
        'neutral': get_token_id('neutral'), # 7974 
        'sentiment': get_token_id('sentiment')
    }

    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1) # roberta doesn't have token_type_ids
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4

    targets_seq = [0.0]*targets_start + [1.0]*(targets_end - targets_start + 1) # [!]
    targets_seq += [0.0]*(len(input_ids) - len(targets_seq))

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

        targets_seq = targets_seq + [0.0]*padding_length # [!]

    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'targets_seq': targets_seq, # [!]
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets,
    }


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text, tokenizer, max_seq_len, model_type):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = tokenizer
        self.max_len = max_seq_len
        self.model_type = model_type
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        if self.model_type == 'roberta':
            data = process_data_roberta(
                self.tweet[item],
                self.selected_text[item],
                self.sentiment[item],
                self.tokenizer,
                self.max_len
            )
        else:
            data = process_data(
                self.tweet[item],
                self.selected_text[item],
                self.sentiment[item],
                self.tokenizer,
                self.max_len
            )
        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'targets_seq': torch.tensor(data["targets_seq"], dtype=torch.float), # [!]
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'sentiment_': SENT_TO_IDX[data["sentiment"]],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long),
        }
