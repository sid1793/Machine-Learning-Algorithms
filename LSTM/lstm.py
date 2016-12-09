"""
Simple implementation of a one layer LSTM with softmax layer for prediction
"""
import numpy as np
from collections import defaultdict

# seed np random generator
np.random.seed(1234)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

class LstmCell(object):

    def __init__(self, input_dim, num_hidden):
        # Model meta-data
        self.input_dim = input_dim
        self.num_hidden = num_hidden

        # Initialize model parameters
        self.Wix = np.random.randn(num_hidden, input_dim)*0.01
        self.Wih = np.random.randn(num_hidden, input_dim)*0.01
        self.Wox = np.random.randn(num_hidden, input_dim)*0.01
        self.Woh = np.random.randn(num_hidden, input_dim)*0.01
        self.Wfx = np.random.randn(num_hidden, input_dim)*0.01
        self.Wfh = np.random.randn(num_hidden, input_dim)*0.01
        self.Wgx = np.random.randn(num_hidden, input_dim)*0.01
        self.Wgh = np.random.randn(num_hidden, input_dim)*0.01
        self.Why = np.random.randn(input_dim, num_hidden)*0.01 # input_dim = output_dim
        self.bi = np.zeros((num_hidden,1))
        self.bo = np.zeros((num_hidden,1))
        self.bf = np.zeros((num_hidden,1))
        self.bg = np.zeros((num_hidden,1))
        self.by = np.zeros((input_dim,1))

    def forward(self, inputs, targets, h_prev, c_prev):
        # Forward pass of lstm
        cache = defaultdict(lambda x: defaultdict)
        cache['h'][-1] = np.copy(h_prev)
        cache['c'][-1] = np.copy(c_prev)
        loss = 0

        for t in xrange(len(inputs)):
            cache['i'][t] = sigmoid(np.dot(self.Wix, inputs[t]) + np.dot(self.Wih, cache['h'][t-1]) + self.bi) # i(t)
            cache['o'][t] = sigmoid(np.dot(self.Wox, inputs[t]) + np.dot(self.Woh, cache['h'][t-1]) + self.bo) # o(t)
            cache['f'][t] = sigmoid(np.dot(self.Wfx, inputs[t]) + np.dot(self.Wfh, cache['h'][t-1]) + self.bf) # f(t)
            cache['g'][t] = np.tanh(np.dot(self.Wgx, inputs[t]) + np.dot(self.Wgh, cache['h'][t-1]) + self.bg) # g(t)
            cache['c'][t] = cache['g'][t] * cache['i'][t] + cache['c'][t-1] * cache['f'][t] # c(t)
            cache['h'][t] = cache['c'][t] * cache['o'][t] # h(t)
            cache['y'][t] = np.dot(self.Why, cache['h'][t]) + self.by # unnormalized log probabilities
            cache['p'][t] = np.exp(cache['y'][t]) / np.sum(np.exp(cache['y'][t])) # softmax for prediction
            loss += -np.log(cache['p'][t][targets[t]])

        return cache, loss

    def backward(self, cache, targets):
        # Backward pass to compute gradients
        dWix, dWih, dWox, dWoh, dWfx, dWfh, dWgx, dWgh = [np.zeros_like(self.Wix) for i in range(8)]
        dbi, dbo, dbf, dbg = [np.zeros_like(self.bi) for i in range(4)]
        dWhy, dby = np.zeros_like(self.Why), np.zeros_like(self.by)
        dhnext, dcnext = np.zeros_like(cache['h'][0]), np.zeros_like(cache['s'][0])

        for t in reverse(xrange(len(targets))):
            dy = cache['p'][t]
            dy[target] -= 1 # backprop into y
            dby += dy # bias of y
            dWhy += np.dot(dy, cache['h'][t].T)
            dh = np.dot(self.Why.T, dy) + dhnext # backprop into h from branches

            # Backprop into c, i, o, g and f
            dc = dh * cache['o'][t] + dcnext # backprop into c from two branches
            do = dh * cache['c'][t]
            di = dg * cache['g'][t]
            dg = di * cache['i'][t]
            df = dc * cache['c'][t-1]

            # Backprop into non-linearities
            diraw = (1 - cache['i'][t]) * cache['i'][t] * di
            doraw = (1 - cache['o'][t]) * cache['o'][t] * do
            dfraw = (1 - cache['f'][t]) * cache['f'][t] * df
            dgraw = (1 - cache['g'][t]**2) * dg

            # Backprop into model params
            dWix += np.dot(diraw, cache['x'][t].T)
            dWih += np.dot(diraw, cache['h'][t-1].T)
            dWox += np.dot(doraw, cache['x'][t].T)
            dWoh += np.dot(doraw, cache['h'][t-1].T)
            dWfx += np.dot(dfraw, cache['x'][t].T)
            dWfh += np.dot(dfraw, cache['h'][t-1].T)
            dWgx += np.dot(dgraw, cache['x'][t].T)
            dWgh += np.dot(dgraw, cache['h'][t-1].T)
            dbi += diraw
            dbo += doraw
            dbf += dfraw
            dbg += dgraw

            # Gradient for next iteration
            dhnext = np.sum(np.dot(x.T, y) for x,y in zip([self.Wih, self.Woh, self.Wfh, self.Wgh],
                                                          [diraw, doraw, dfraw, dhraw]))
            dcnext = dc * cache['f'][t]

            # Return gradients
            return dWix, dWih, dWox, dWoh, dWfx, dWfh, dWgx, dWgh, dWhy, dbi, dbo, dbf, dbg, dby
