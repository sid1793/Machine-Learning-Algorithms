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
    for i in data:
        x = np.zeros((vocab_len,1))
        x[char2idx[i]] = 1
        input_data.append(x)
        targets.append(x)

    return input_data, targets, char2idx, idx2char, vocab_len

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def sample_chars(model, h_prev, c_prev, first_word, length):
    """
    Sample chars from model 
    """
    x = first_word
    seq = []
    for i in range(length):
        i = sigmoid(np.dot(model.Wix, x) + np.dot(model.Wih, h_prev) + model.bi)
        o = sigmoid(np.dot(model.Wox, x) + np.dot(model.Woh, h_prev) + model.bo)
        f = sigmoid(np.dot(model.Wfx, x) + np.dot(model.Wfh, h_prev) + model.bf)
        g = np.tanh(np.dot(model.Wgx, x) + np.dot(model.Wgh, h_prev) + model.bg)
        c = g * i + c_prev * f
        h = c * o
        y = np.dot(model.Why, h) + model.by
        p = np.exp(y) / np.sum(np.exp(y))
        pred = np.random.choice(range(len(x)), p = p.ravel())
        seq.append(pred)
        h_prev = h
        c_prev = c
        x = np.zeros((len(x),1))
        x[pred] = 1

    return seq
