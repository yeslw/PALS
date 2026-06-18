# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import os
import numpy as np
import pandas as pd
import random
import torch
from datasets import load_dataset, Dataset

WIKITEXT2_TRAIN_PATH = "/2T/zhuhe/data/wikitext-2-raw-v1/train-00000-of-00001.parquet"
WIKITEXT2_TEST_PATH = "/2T/zhuhe/data/wikitext-2-raw-v1/test-00000-of-00001.parquet"
C4_TRAIN_PATH = "/2T/zhuhe/data/C4/train/c4-train.00000-of-01024.json"

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load local train and test datasets from parquet files
    train_data = pd.read_parquet(WIKITEXT2_TRAIN_PATH)
    test_data = pd.read_parquet(WIKITEXT2_TEST_PATH)

    # Convert pandas dataframe to Hugging Face Dataset
    train_data = Dataset.from_pandas(train_data)
    test_data = Dataset.from_pandas(test_data)

    # Encode datasets
    trainenc = tokenizer(" ".join(train_data['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(test_data['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('json', data_files={'train': C4_TRAIN_PATH}, split='train')
    
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(traindata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    
    # traindata = load_from_disk("datasets/ptb/train")
    # testdata = load_from_disk("datasets/ptb/test")

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)

