from typing import Iterable, Union, List
from prenlp.data import IMDB
from tqdm import tqdm
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import TensorDataset


DATASETS_CLASSES = {'imdb': IMDB}
DATASETS_SELFMADE = {
    'orgs': [
        np.load('./.data/orgs/train.npy'),
        np.load('./.data/orgs/val.npy')
    ]
}


class InputExample:
    """
    A single training/test sample for text classification.
    """
    def __init__(self, text: str, label: str):
        self.text = text
        self.label = label


class InputFeatures:
    """
    A single set of features (i.e., ids) of data.
    """
    def __init__(self, input_ids: List[int], label_id: int):
        self.input_ids = input_ids
        self.label_id = label_id


def convert_examples_to_features(examples: List[InputExample],
                                 label_dict: dict,
                                 tokenizer) -> List[InputFeatures]:
    features = []
    print('Converting samples to features...')
    for i, example in tqdm(enumerate(examples)):
        input_ids = tokenizer.encode([example.text])[0]
        label_id = label_dict.get(example.label)
        
        feature = InputFeatures(input_ids, label_id)
        features.append(feature)

    return features


def create_examples(dataset_name: str,
                    tokenizer,
                    dataset_type: str = 'packed',
                    mode: str = 'train') -> Iterable[Union[List[InputExample], dict]]:
    if dataset_type == 'packed':
        if mode == 'train':
            dataset = DATASETS_CLASSES[dataset_name]()[0]
        elif mode == 'test':
            dataset = DATASETS_CLASSES[dataset_name]()[1]
    elif dataset_type == 'selfmade':
        if mode == 'train':
            dataset = DATASETS_SELFMADE[dataset_name][0]
        elif mode == 'test':
            dataset = DATASETS_SELFMADE[dataset_name][1]
    else:
        raise NotImplementedError

    # all data as InputExample class
    examples = []
    for text, label in dataset:
        example = InputExample(text, label)
        examples.append(example)
    
    # find out all labels
    labels = sorted(list(set([example.label for example in examples])))
    # encode the labels
    label_dict = {label: i for i, label in enumerate(labels)}
    print('[{}]\tLabel dictionary:\t{}'.format(mode, label_dict))
    with open(f"./label_dict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
        f.writelines(label_dict)

    # encode the contents
    features = convert_examples_to_features(examples, label_dict, tokenizer)
    
    all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    all_label_ids = torch.tensor([feature.label_id for feature in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_label_ids)

    return dataset


# #### TEST MODULE ####
# # format:
# # [['text', 'label'], ['text', 'label'], ...]
# data_train = DATASETS_CLASSES['imdb']()[0]
# data_test = DATASETS_CLASSES['imdb']()[1]


# examples = []
# for text, label in data_train:
#     example = InputExample(text, label)
#     examples.append(example)

# labels = sorted(list(set([example.label for example in examples])))
# label_dict = {label: i for i, label in enumerate(labels)}

# from tokenization_self import PretrainedTokenizer
# tokenizer = PretrainedTokenizer(
#     './models/tokenizer.model',
#     max_seq_len=64,
#     padding_strategy='max_length',
#     pad_id=0,
# )

# features = []
# for i, example in tqdm(enumerate(examples)):
#     input_ids = tokenizer.encode([example.text])[0]
#     label_id = label_dict.get(example.label)
    
#     feature = InputFeatures(input_ids, label_id)
#     features.append(feature)

# features[0].input_ids

# all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
# all_label_ids = torch.tensor([feature.label_id for feature in features], dtype=torch.long)
# #####################