import numpy as np
from lstm import LstmCell
from util import preprocess_char_text_data, sample_chars

def train_char(model, learning_rate, input, targets, batch_size, idx2char):
    """
    Train char-LSTM cell using SGD
    """
    #for i in xrange(iterations):
    idx = 0
    n = 0
    while True:

        if idx + batch_size + 1 >= len(input) or idx == 0:
            # reset memory state of LSTM
            h_prev = np.zeros((model.num_hidden, 1))
            c_prev = np.zeros((model.num_hidden, 1))
            idx = 0

        # forward step
        cache, loss = model.forward(input[idx:idx+batch_size], targets[idx+1:idx+batch_size+1], h_prev, c_prev)

        if n % 100 == 0:
            print 'Loss at iteration {} is {}'.format(n, loss)
            sam_seq = sample_chars(model, h_prev, c_prev, input[0], 200)
            sent = ''.join(idx2char[i] for i in sam_seq)
            print '-----\n {} \n----'.format(sent)

        # backward pass to get gradients
        dWix, dWih, dWox, dWoh, dWfx, dWfh, dWgx, dWgh, dWhy, dbi, dbo, dbf, dbg, dby = model.backward(cache, input[idx:idx+batch_size], targets[idx+1:idx+batch_size+1])

        # update model params
        model.Wix -= learning_rate * dWix
        model.Wih -= learning_rate * dWih
        model.Wox -= learning_rate * dWox
        model.Woh -= learning_rate * dWoh
        model.Wfx -= learning_rate * dWfx
        model.Wfh -= learning_rate * dWfh
        model.Wgx -= learning_rate * dWgx
        model.Wgh -= learning_rate * dWgh
        model.Why -= learning_rate * dWhy
        model.bi -= learning_rate * dbi
        model.bo -= learning_rate * dbo
        model.bf -= learning_rate * dbf
        model.bg -= learning_rate * dbg
        model.by -= learning_rate * dby

        # set h_prev and c_prev
        h_prev = cache['h'][batch_size-1]
        c_prev = cache['c'][batch_size-1]

        # increment data pointer and iteration number
        idx += batch_size
        n += 1

def main():
    # get text data
    input, targets, char2idx, idx2char, vocab_len = preprocess_char_text_data('input.txt')

    # train model
    model = LstmCell(vocab_len, 100)
    #cache, loss = model.forward(input[:10], targets[1:11], np.zeros((100,1)), np.zeros((100,1)))
    #print len(cache['p'][9])
    train_char(model, 0.1, input, targets, 25, idx2char)

if __name__ == "__main__":
    main()
