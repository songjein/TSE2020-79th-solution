"""Build Model"""
import torch

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer

from transformers import BertModel, BertConfig
from transformers import RobertaModel, RobertaConfig
from transformers import ElectraModel, ElectraConfig, ElectraTokenizerFast


from transformers import AutoModelForQuestionAnswering
from transformers.modeling_bert import BertPreTrainedModel


import config as cfg


def init_tokenizer(model_type):
    """토크나이저 사용 전에 반드시 호출"""

    model_path_dict = cfg.get_model_path(model_type)

    print(model_path_dict)

    if model_type in ['bert', 'electra']:
        tokenizer = BertWordPieceTokenizer(
            model_path_dict['tokenizer'],
            lowercase=True
        )
    elif model_type == 'roberta':
        tokenizer = ByteLevelBPETokenizer(
            vocab_file=model_path_dict['tokenizer']['vocab_file'],
            merges_file=model_path_dict['tokenizer']['merges_file'],
            lowercase=True,
            add_prefix_space=True
        )
    return tokenizer


def _get_bert(
    model_type,
    model_path_dict):
    if model_type == 'bert':
        config = BertConfig.from_pretrained(
            model_path_dict['config']
        )
        config.output_hidden_states = True
        bert = BertModel.from_pretrained(
            model_path_dict['model'],
            config=config)
    elif model_type == 'electra':
        config = ElectraConfig.from_pretrained(
            model_path_dict['config'])
        config.output_hidden_states = True
        bert = ElectraModel.from_pretrained(
            model_path_dict['model'],
            config=config)
    elif model_type == 'roberta':
        config = RobertaConfig.from_pretrained(
            model_path_dict['config'])
        config.output_hidden_states = True
        bert = RobertaModel.from_pretrained(
            model_path_dict['model'],
            config=config)
    return bert, config


def _get_hidden(bert, model_type, input_ids, attention_mask, token_type_ids):
    if model_type == 'bert':
        hidden_states, _, hidden = bert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
    elif model_type == 'electra':
        hidden_states, hidden = bert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
    elif model_type == 'roberta':
        hidden_states, _, hidden = bert(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             token_type_ids=token_type_ids)
    return hidden


class SentimentExtractor(torch.nn.Module):
    
    def __init__(self, model_type='electra', max_seq_len=128, dropout_rate=0.1, last_n_layers=2):
        super(SentimentExtractor, self).__init__()

        model_path_dict = cfg.get_model_path(model_type)

        self.model_type = model_type.lower()
        self.last_n_layers = last_n_layers

        self.bert, self.config = _get_bert(self.model_type, model_path_dict)
        bert_hidden_dim = self.bert.config.hidden_size

        self.fc_for_idx = torch.nn.Linear(bert_hidden_dim*last_n_layers, 2)
        self.fc_for_seq = torch.nn.Linear(bert_hidden_dim*last_n_layers, 1)

        self.bn_1 = torch.nn.BatchNorm1d(max_seq_len)
        self.bn_2 = torch.nn.BatchNorm1d(max_seq_len)

        self.dropout = torch.nn.Dropout(dropout_rate)

        torch.nn.init.normal_(self.fc_for_idx.weight, std=0.02)
        torch.nn.init.normal_(self.fc_for_seq.weight, std=0.02)
        
    def forward(self, input_ids, attention_mask, token_type_ids):

        hidden = _get_hidden(self.bert, self.model_type, input_ids, attention_mask, token_type_ids)

        hidden_concat = torch.cat(
            tuple(
                hidden[-(idx+1)] for idx in range(self.last_n_layers)
            ),
            dim=-1
        )
        hidden_concat = self.bn_1(hidden_concat)
        hidden_concat = self.dropout(hidden_concat) # (batch, seq, 2)

        seq_logits = self.fc_for_seq(hidden_concat) # (batch, seq, 1)
        seq_logits = self.bn_2(seq_logits)
        seq_attn = torch.sigmoid(seq_logits)

        _seq_logits = seq_logits.squeeze(-1) # (batch, seq)

        hidden_attn = hidden_concat*seq_attn
        idx_logits = self.fc_for_idx(hidden_attn)
        idx_logits_splited = torch.split(idx_logits, split_size_or_sections=1, dim=-1) # (batch, seq, 1), (batch, seq, 1)

        start_logits = idx_logits_splited[0].squeeze(-1)
        end_logits = idx_logits_splited[1].squeeze(-1)

        return start_logits, end_logits, _seq_logits




if __name__ == '__main__':
    a = torch.tensor([
        [
            [1,1,1,1],
            [2,2,2,2],
        ],
        [
            [3,3.1,3.2,3.3],
            [4,4.1,4.2,4.3],
        ],
        [
            [5,5,5,5],
            [6,6,6,6],
        ]
    ]) # (3b, 2t, 4h)

    a = torch.tensor([
        [
            [1,1,10],
            [2,2,9],
            [3,3,11],
        ] 
    ]) # (3b, 3tokens) 

    b = torch.tensor([
        [0.1, 0.5, 0.1],
        [0.3, 0.7, 0.2],
        [0.8, 0.2, 0.1],
    ]) # (3b, 3tokens)
    # b = b.unsqueeze(-1)

    print(a.size())
    print(a)
    
    print()     
    print(b.size())
    print(b)

    print()
    print(a*b)
    
    x = torch.tensor([
        [
            [0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
            [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1],
        ],
    ])

    y = torch.tensor([
        [0, 0, 1, 1, 1, 0, 0, 0]
    ])

    print(torch.nn.CrossEntropyLoss()(x, y))

