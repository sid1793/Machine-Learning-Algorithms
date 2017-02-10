import numpy as np
import tensorflow as tf

class PGAgent(object):

    def __init__(self, observation_space, action_space):
        """
        Build computation graph for stochastic policy network and corresponding
        functions to compute gradients and updates
        """
        raise NotImplementedError

    def take_action(self, obs):
        """
        Perform action by sampling from probability distribution defined by
        policy network over action space
        """
        raise NotImplementedError

    def update_policy(self, advantage_vals):
        """
        Computes gradient of the score function estimator wrt to policy params
        and updates params
        """
        raise NotImplementedError
