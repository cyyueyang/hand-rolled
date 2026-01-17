from collections import defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq

    return pairs

def merge_vocab(pair, vocab_in):
    vocab_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab_in:
        w_out = word.replace(bigram, replacement)
        vocab_out[w_out] = vocab_in[word]
    return vocab_out

def train_bpe(vocab, num_merges):
    merges = {}
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if len(pairs) == 0:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges[best_pair] = i

    return merges, vocab

if __name__ == '__main__':
    vocab = {
        'l o w</w>': 5,
        'l o w e r</w>': 2,
        'n e w e s t</w>': 6,
        'w i d e s t</w>': 3,
    }
    merges, vocab = train_bpe(vocab, 10)
    print(merges)
    print(vocab)



