import os
import json
import re

def read_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab
def real_merges(merges_path):
    with open(merges_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    merges = []
    for line in lines:
        pair = line.strip().split()
        if len(pair) == 2:
            merges.append(pair)
    return merges