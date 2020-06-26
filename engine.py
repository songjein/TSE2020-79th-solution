"""funcs for training"""
import time 

import numpy as np
import torch
from tqdm import tqdm

import util
import config as cfg


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def do_train_at(k_fold_i,
                epoch_i,
                train_data_loader,
                model,
                es,
                optimizer,
                scheduler,
                max_seq_len,
                device='cpu',
                adv=False,
                use_dist_loss=False,
                use_seq_loss=False,
                ce_smoothing=False,
                ce_smoothing_ratio=0.2):
    """do train at (k_fold_i, epoch_i)"""
    
    losses = util.AverageMeter()
    jaccards = util.AverageMeter()
    
    start_accs = util.AverageMeter()
    end_accs = util.AverageMeter()
    
    num_train_batch = len(train_data_loader)
    
    t0 = time.time()
    
    fgm = FGM(model)

    # [START] for step, batch in enumerate(train_data_loader): (
    for step, batch in enumerate(tqdm(train_data_loader, desc="Train", ncols=80)):
        
        ids = batch["ids"]
        token_type_ids = batch["token_type_ids"]
        mask = batch["mask"]
        sentiment = batch["sentiment"]
        targets_senti = batch["sentiment_"]
        orig_selected = batch["orig_selected"]
        orig_tweet = batch["orig_tweet"]
        targets_start = batch["targets_start"]
        targets_end = batch["targets_end"]
        targets_seq = batch["targets_seq"]
        offsets = batch["offsets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        targets_seq = targets_seq.to(device, dtype=torch.float)
        
        model.train()
        model.zero_grad() 

        start_logits, end_logits, seq_logits = model(ids,
                                                     token_type_ids=token_type_ids,
                                                     attention_mask=mask)

        if ce_smoothing:
            start_loss = util.SmoothCrossEntropyLoss(smoothing=ce_smoothing_ratio)(start_logits, targets_start)
            end_loss = util.SmoothCrossEntropyLoss(smoothing=ce_smoothing_ratio)(end_logits, targets_end)
        else:
            start_loss = torch.nn.CrossEntropyLoss()(start_logits, targets_start)
            end_loss = torch.nn.CrossEntropyLoss()(end_logits, targets_end)

        idx_loss = (start_loss+end_loss) # /2 # [TODO]

        total_loss = idx_loss

        if use_dist_loss:
            dist_loss = util.dist_loss(
                start_logits, end_logits,
                torch.tensor(targets_start), torch.tensor(targets_end),
                device, max_seq_len,
                scale=1.0)
            total_loss = total_loss+dist_loss

        if use_seq_loss:
            seq_loss = torch.nn.BCEWithLogitsLoss()(seq_logits, targets_seq)
            total_loss = (total_loss+seq_loss)/2
        
        total_loss.backward()

        if adv and epoch_i > 0:
            fgm.attack()
            _start_logits, _end_logits, _seq_logits = model(ids,
                                                         token_type_ids=token_type_ids,
                                                         attention_mask=mask)
            if ce_smoothing:
                _start_loss = util.SmoothCrossEntropyLoss(smoothing=ce_smoothing_ratio)(_start_logits, targets_start)
                _end_loss = util.SmoothCrossEntropyLoss(smoothing=ce_smoothing_ratio)(_end_logits, targets_end)
            else:
                _start_loss = torch.nn.CrossEntropyLoss()(_start_logits, targets_start)
                _end_loss = torch.nn.CrossEntropyLoss()(_end_logits, targets_end)

            _idx_loss = (_start_loss+_end_loss)

            total_loss_adv = _idx_loss
            total_loss_adv.backward()
            fgm.restore()

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        model.eval() # [!]
        
        losses.update(total_loss.item(), ids.size(0))
        
        outputs_start = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
        targets_start = targets_start.cpu().detach().numpy()
        targets_end = targets_end.cpu().detach().numpy()
        
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = util.calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            jaccard_scores.append(jaccard_score)

        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        
        start_acc = util.flat_accuracy(outputs_start, targets_start)
        start_accs.update(start_acc, ids.size(0))
        
        end_acc = util.flat_accuracy(outputs_end, targets_end)
        end_accs.update(end_acc, ids.size(0))

    # [END] for step, batch in enumerate(train_data_loader): )

    return {
        'time': util.format_time(time.time() - t0),
        'loss': losses.avg,
        'jaccard': jaccards.avg,
        'start_acc': start_accs.avg,
        'end_acc': end_accs.avg,
    }


def do_validation_at(val_data_loader,
                     model,
                     max_seq_len,
                     device='cpu',
                     use_dist_loss=False,
                     use_seq_loss=False,
                     ce_smoothing=False,
                     ce_smoothing_ratio=0.2):
    """do validation at (k_fold_i, epoch_i)"""
    model.eval()
    
    t0 = time.time()

    losses = util.AverageMeter()
    jaccards = util.AverageMeter()
    
    start_accs = util.AverageMeter()
    end_accs = util.AverageMeter()

    for batch in tqdm(val_data_loader, desc="Valid", ncols=80):
        
        ids = batch["ids"]
        token_type_ids = batch["token_type_ids"]
        mask = batch["mask"]
        sentiment = batch["sentiment"]
        targets_senti = batch["sentiment_"]
        orig_selected = batch["orig_selected"]
        orig_tweet = batch["orig_tweet"]
        targets_start = batch["targets_start"]
        targets_end = batch["targets_end"]
        targets_seq = batch["targets_seq"] # [!]
        offsets = batch["offsets"].numpy()
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        targets_seq = targets_seq.to(device, dtype=torch.float)
        
        with torch.no_grad():
            start_logits, end_logits, seq_logits = model(ids,
                                                         token_type_ids=token_type_ids,
                                                         attention_mask=mask)
        
        if ce_smoothing:
            start_loss = util.SmoothCrossEntropyLoss(smoothing=ce_smoothing_ratio)(start_logits, targets_start)
            end_loss = util.SmoothCrossEntropyLoss(smoothing=ce_smoothing_ratio)(end_logits, targets_end)
        else:
            start_loss = torch.nn.CrossEntropyLoss()(start_logits, targets_start)
            end_loss = torch.nn.CrossEntropyLoss()(end_logits, targets_end)

        total_loss_ = (start_loss.item()+end_loss.item()) # /2 # [TODO]

        if use_dist_loss:
            dist_loss = util.dist_loss(
                start_logits, end_logits,
                torch.tensor(targets_start), torch.tensor(targets_end),
                device, max_seq_len)
            total_loss_ += dist_loss.item()

        if use_seq_loss:
            seq_loss = torch.nn.BCEWithLogitsLoss()(seq_logits, targets_seq)
            total_loss_ = (total_loss_+seq_loss.item())/2
 
        losses.update(total_loss_, ids.size(0))
        
        outputs_start = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
        targets_start = targets_start.cpu().detach().numpy()
        targets_end = targets_end.cpu().detach().numpy()

        jaccard_scores = []
        for i, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[i]
            tweet_sentiment = sentiment[i]
            jaccard_score, _ = util.calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[i, :]),
                idx_end=np.argmax(outputs_end[i, :]),
                offsets=offsets[i]
            )
            jaccard_scores.append(jaccard_score) 
        
        jaccards.update(np.mean(jaccard_scores), ids.size(0))

        start_acc = util.flat_accuracy(outputs_start, targets_start)
        start_accs.update(start_acc, ids.size(0))
        
        end_acc = util.flat_accuracy(outputs_end, targets_end)
        end_accs.update(end_acc, ids.size(0))

    return {
        'time': util.format_time(time.time()-t0),
        'jaccard': jaccards.avg,
        'loss': losses.avg,
        'start_acc': start_accs.avg,
        'end_acc': end_accs.avg,
    }


def get_hard_examples(train_data_loader,
                      model,
                      device='cpu'):
    """make hard examples"""
    model.eval()

    hard_examples = []

    for step, batch in enumerate(train_data_loader):

        ids = batch["ids"]
        token_type_ids = batch["token_type_ids"]
        mask = batch["mask"]
        sentiment = batch["sentiment"]
        targets_senti = batch["sentiment_"]
        orig_selected = batch["orig_selected"]
        orig_tweet = batch["orig_tweet"]
        targets_start = batch["targets_start"]
        targets_end = batch["targets_end"]
        targets_seq = batch["targets_seq"] # [!]
        offsets = batch["offsets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        targets_seq = targets_seq.to(device, dtype=torch.long) # [!]

        model.zero_grad()

        with torch.no_grad():
            start_logits, end_logits, seq_logits = model(ids,
                                                         token_type_ids=token_type_ids,
                                                         attention_mask=mask)

        outputs_start = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
        targets_start = targets_start.cpu().detach().numpy()
        targets_end = targets_end.cpu().detach().numpy()

        jaccard_scores = []
        _hard_examples = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = util.calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            if jaccard_score < 0.5: # [!]
                _hard_examples.append((tweet, selected_tweet, tweet_sentiment))
            jaccard_scores.append(jaccard_score)
        hard_examples += _hard_examples

    return hard_examples
