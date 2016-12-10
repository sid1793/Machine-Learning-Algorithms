import numpy as np
from lstm import LstmCell
from util import preprocess_char_text_data

def train(model, learning_rate, input, target, iterations, batch_size):
    """
    Train LSTM cell using SGD
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
        cache, loss = model.forward(input[idx:idx+batch_size], target[idx+1:idx+batch_size+1], h_prev, c_prev)

        if n % 100 == 0:
            print 'Loss at iteration {} is {}'.format(idx, loss)

        # backward pass to get gradients
        dWix, dWih, dWox, dWoh, dWfx, dWfh, dWgx, dWgh, dWhy, dbi, dbo, dbf, dbg, dby = model.backward(cache, input[idx:idx+batch_size], target[idx+1:idx+batch_size+1])

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
    input, target, char2idx, idx2char = preprocess_char_text_data('input.txt')

    # train model
    model = LstmCell(len(char2idx), 100)
    #cache, loss = model.forward(input[:10], target[1:11], np.zeros((100,1)), np.zeros((100,1)))
    #print len(cache['p'][9])
    train(model, 0.1, input, target, 100, 25)

if __name__ == "__main__":
    main()
