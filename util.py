"""utils for preproc & training"""
import re
import datetime

import numpy as np
import torch
import pandas as pd

import torch.nn.functional as F

from sklearn.utils import resample
from torch.nn.modules.loss import _WeightedLoss
    
import config as cfg


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        
class EarlyStopping:
    def __init__(self, patience=20, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('    Early Stopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('    Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def argmax(logits):
    pred_flat = np.argmax(logits, axis=1).flatten()
    return pred_flat


def get_jaccard_score(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_jaccard_score(
    original_tweet, 
    target_string, 
    sentiment_val, 
    idx_start, 
    idx_end,
    offsets,
    verbose=False):
    """calculate score againt GT & restore predicted selected_text using start/end index"""
    
    if idx_end < idx_start:
        idx_end = idx_start
        # [TODO] 어떤 놈은 그냥 text를 리턴
    
    filtered_output  = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    # if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
    if len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    jac = get_jaccard_score(target_string.strip(), filtered_output.strip())
    return jac, filtered_output


def split_valid_points(num_batch, valid_split):
    """get points for multiple validations in an epoch (ex. [200, 400, 600, 800] with num_batch=800, valid_split=4)"""
    if valid_split == 0:
        return [num_batch-1]

    unit = num_batch//valid_split
    points = [unit-1]
    for i in range(valid_split-1):
        points.append(points[-1]+unit)
    points[-1] = num_batch-1
    return points


def dist_between(start_logits, end_logits, device='cpu', max_seq_len=128):
    """get dist btw. pred & ground_truth"""

    linear_func = torch.tensor(np.linspace(0, 1, max_seq_len, endpoint=False), requires_grad=False)
    linear_func = linear_func.to(device)

    start_pos = (start_logits*linear_func).sum(axis=1)
    end_pos = (end_logits*linear_func).sum(axis=1)

    diff = end_pos-start_pos

    return diff.sum(axis=0)/diff.size(0)


def dist_loss(start_logits, end_logits, start_positions, end_positions, device='cpu', max_seq_len=128, scale=1.0):

    start_logits = torch.nn.Softmax(1)(start_logits) # (batch, max_seq_len)
    end_logits = torch.nn.Softmax(1)(end_logits)

    start_one_hot = torch.nn.functional.one_hot(start_positions, num_classes=max_seq_len).to(device)
    end_one_hot = torch.nn.functional.one_hot(end_positions, num_classes=max_seq_len).to(device)

    pred_dist = dist_between(start_logits, end_logits, device, max_seq_len) # if predicted well -> positive, else -> negative
    gt_dist = dist_between(start_one_hot, end_one_hot, device, max_seq_len) # always positive
    diff = (gt_dist-pred_dist)

    rev_diff_squared = 1-torch.sqrt(diff*diff) # diff 부호를 없애주고, 차이가 적을 수록 1에 가깝게 하기 위해 1에서 뺌
    loss = -torch.log(rev_diff_squared) # 0에 가까울 수록 무한대에 가까운 로스, 1에 가까울 수록 0에 가까운 로스  

    return loss*scale


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.20):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            l_1 = targets-1
            r_1 = targets+1
            prob = smoothing/2.0
            targets = torch.zeros(size=(targets.size(0), n_classes), device=targets.device) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing) \
                .scatter_(1, l_1.data.unsqueeze(1), prob) \
                .scatter_(1, r_1.data.unsqueeze(1), prob)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1), self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()
        return loss


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def find_punct(text):
    line = re.findall(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', text)
    string="".join(line)
    return list(string)


def cleaning(text, selected_text, sentiment):
    """애매하게 잘린 답안의 경우 잘린 부분을 포함한 단어로 확장시켜준다."""
    if sentiment == 'neutral':
        return selected_text
    orig_selected_text = selected_text
    start = text.find(selected_text)
    end = start+len(selected_text)-1

    # 답안의 맨 뒤가 불완전하게 잘려 있으면 확장. 단, 구두점이라면 확장하지 않음
    if end+1 < len(text) and (text[end+1] != ' '):
        if len(find_punct(text[end+1])) > 0:
            pass
        elif len(find_punct(text[end])) > 0:
            pass
        else:
            # 우로 확장하기
            while end+1 < len(text) and text[end+1] != ' ' and len(find_punct(text[end+1])) == 0:
                selected_text += text[end+1]
                end += 1

    # 답안의 앞부분이 불완전하게 잘려있으면 확장. 단, 구두점이라면 확장하지 않음
    if start > 0 and text[start-1] != ' ':

        # 공통적으로, 답안 앞부분의 구두점은 무조건 제거
        removed = False
        while len(selected_text) > 0 and len(find_punct(selected_text[0])) > 0:
            selected_text = selected_text[1:]
            removed = True

        if removed:
            pass
        elif len(find_punct(text[start-1])) > 0:
            pass
        else:
            # 좌로 확장하기
            while start > 0 and text[start-1] != ' ' and len(find_punct(text[start-1])) == 0:
                selected_text = text[start-1] + selected_text
                start -= 1
    if len(selected_text) == 0:
        return orig_selected_text

    return selected_text


def equalize_samples(df, seed=42):
    neu_len = len(df[df.sentiment=='neutral'])
    pos_len = len(df[df.sentiment=='positive'])
    neg_len = len(df[df.sentiment=='negative'])

    samples = min(neu_len, pos_len, neg_len)

    neu_df = df[df.sentiment=='neutral']
    pos_df = df[df.sentiment=='positive']
    neg_df = df[df.sentiment=='negative']

    pos_df = resample(pos_df, replace=False, n_samples=samples, random_state=seed)
    neg_df = resample(neg_df, replace=False, n_samples=samples, random_state=seed)
    neu_df = resample(neu_df, replace=False, n_samples=samples, random_state=seed)

    concat_df = pd.concat([pos_df, neg_df, neu_df])

    concat_df = concat_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return concat_df


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def text_change_pre(text, patterns_dict=None):
    if patterns_dict is None:
        patterns_dict = cfg.PATTERNS_DICT
    replace_dict = match_helper(text, patterns_dict, 0, len(text)) # slice index to new word
    reverse_dict = dict() # used to change from output_string to original string
    keys = list(replace_dict.keys())
    keys.sort()
    output = []
    cur = 0
    output_cur = 0
    for key in keys:
        new_word = replace_dict[key]
        output_cur += key[0]-cur
        output.append(text[cur:key[0]])
        output.append(new_word)
        reverse_key = (output_cur, output_cur+len(new_word))
        output_cur += len(new_word)
        reverse_dict[reverse_key] = text[key[0]:key[1]]
        cur = key[1]
    output.append(text[cur:])
    output_string = ''.join(output)
    return output_string, replace_dict, reverse_dict



def match_helper(text, patterns_dict, start, end):
    cur_text = text[start: end]

    for pattern in patterns_dict:
        match = re.search(pattern, cur_text)
        if match is None:
            continue
        else:
            key_start = start + match.start()
            key_end = start + match.end()

            replace_dict = match_helper(text, patterns_dict, start, key_start)
            replace_dict[(key_start, key_end)] = patterns_dict[pattern]
            right_dict = match_helper(text, patterns_dict, key_end, end)
            if right_dict:
                replace_dict.update(right_dict)
            return replace_dict
    # if no pattern found
    return dict()


def text_change_post(text, start_idx, end_idx, reverse_dict):
    keys = list(reverse_dict.keys())
    keys.sort()

    # do not include end_idx for slice
    slice_start = start_idx
    slice_end = end_idx + 1

    ranges = [] # slice index
    tokens = [] # output tokens = 치환된 부분이랑 아닌 부분 구분한 서브스트링 리스트
    is_replace = [] # is reversed from reverse dict
    cur = 0
    for key in keys:
        # key = (pattern_start, pattern_end)
        ranges.append((cur, key[0]))
        tokens.append(text[cur:key[0]])
        is_replace.append(False)

        new_word = reverse_dict[key]
        ranges.append(key)
        tokens.append(new_word)
        is_replace.append(True)

        cur = key[1]

    # remaining string
    ranges.append((cur, len(text)))
    tokens.append(text[cur:])
    is_replace.append(False)

    selected = []
    for i, (token_start, token_end) in enumerate(ranges):
        token = tokens[i]
        if slice_end <= token_start: # no overlap
            continue
        elif token_end <= slice_start: # no overlap
            continue
        elif is_replace[i]: # part of replace word
            selected.append(token) # reverse dict의 replace 단어는 자르지 말기
        else: # part of non replaced word
            # get overlap index for token and selected index
            overlap_start = max(slice_start, token_start)
            overlap_end = min(slice_end, token_end)
            word = text[overlap_start: overlap_end]
            selected.append(word)

    output_string = ''.join(tokens)
    selected_string = ''.join(selected)

    return output_string, selected_string




if __name__ == '__main__':
    import collections

    text = '...I canï¿½t ***** believe tour is ÂAlmost over!!!'
    text = 'Recession hit Veronique Branquinho, she has to quit her company, such a shame!'

    text = 'that`s great!! weee!! visitors!'

    patterns_dict = collections.OrderedDict([
        ("'\\*+", "'t it"),
        ('\\*\\*+', '**'),
        ('!+', '!'),
        ('\\.\\.+',  '..'),
        ('ï¿½', "'"),
        ('ï', "'"),
        ('Â', '.'),
    ])

    print('original:', text)
    new_text, change_dict, reverse_dict = text_change_pre(text, patterns_dict=patterns_dict)
    print('changed:', new_text)

    print(reverse_dict)

# "..I can't ** believe tour is .Almost over!"
# {(0, 3): '..', (8, 11): "'", (13, 18): '**', (35, 36): '.', (47, 50): '!'}
# {(0, 2): '...', (7, 8): 'ï¿½', (10, 12): '*****', (29, 30): 'Â', (41, 42): '!!!'})

    pred_start = 12
    pred_end = 29
    recovered, selected_string = text_change_post(new_text, pred_start, pred_end, reverse_dict)
    print('changed  selected:', new_text[pred_start: pred_end+1])
    print('original selected:', selected_string)

    print('correctly recovered:', recovered == text)
