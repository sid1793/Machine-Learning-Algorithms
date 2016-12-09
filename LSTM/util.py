import numpy as np

def preprocess_char_text_data(filename):
    """
    Data is a sequence of characters
    """
    data = open(filename,'r').read()
    chars = list(set(data))
    data_len, vocab_len = len(data), len(chars)
    char2idx = { ch:i for i, ch in enumerate(chars)} # character to index
    idx2char = { i:ch for i, ch in enumerate(chars)} # index to character

    input_data= []
    targets = []

    # one hot encoding of chars
    for i in data[:10000]:
        x = np.zeros((vocab_len,1))
        x[char2idx[i]] = 1
        input_data.append(x)
        targets.append(x)

    return input_data, targets, char2idx, idx2char
