import numpy as np
import gym

class LinearDetDiscActSpace(object):

    def __init__(self, theta, obs_space, act_space):
        obs_dim = obs_space.shape[0]
        act_num = act_space.n
        assert(len(theta)) == (obs_dim + 1) * act_num
        self.W = theta[ : obs_dim * act_num].reshape(obs_dim, act_num)
        self.b = theta[obs_dim * act_num : ].reshape(1, act_num)

    def take_action(self, ob):
        return np.argmax(ob.dot(self.W) + self.b)

class LinearDetContActSpace(object):

    def __init__(self, theta, obs_space, act_space):
        self.act_space = act_space
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        assert(len(theta)) == (obs_dim + 1) * act_dim
        self.W = theta[ : obs_dim * act_dim].reshape(obs_dim, act_dim)
        self.b = theta[obs_dim * act_dim : ]

    def take_action(self, ob):
        return np.clip(ob.dot(self.W) + self.b, self.act_space.low, self.act_space.high)
