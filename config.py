import os
import datetime

def get_model_path(model_type='roberta'):

    model_base_path = {
        'bert': './bert-squad',
        'electra': './electra-large',
        'roberta': './roberta-squad',
    }

    if model_type in ['bert', 'electra']:
        model_path_dict = {
            'tokenizer': os.path.join(model_base_path[model_type],'vocab.txt'),
            'config': os.path.join(model_base_path[model_type],'config.json'),
            'model': os.path.join(model_base_path[model_type],'pytorch_model.bin'),
        }

    elif model_type == 'roberta':
        model_path_dict = {
            'tokenizer': {
                'vocab_file': os.path.join(model_base_path[model_type],'vocab.json'),
                'merges_file': os.path.join(model_base_path[model_type],'merges.txt'),
            },
            'config': os.path.join(model_base_path[model_type],'config.json'),
            'model': os.path.join(model_base_path[model_type],'pytorch_model.bin'),
        }
    
    return model_path_dict

def make_kfold_output_dir(output_base, memo, model_type, lucky_seed):
    dt = datetime.datetime.now()
    output_dir = os.path.join(
        output_base,
        '{}_{}_{}_{}_{}'.format(memo, model_type, lucky_seed, dt.month, dt.day))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir 
