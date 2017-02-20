import numpy as np
import tensorflow as tf
import math

class PGAgent(object):

    def __init__(self, observation_space, action_space, hidden_dim=100):
        """
        Build computation graph for stochastic policy network and corresponding
        functions to compute gradients and updates
        """
        self.act_dim = act_dim

        # Placeholder for number of steps
        N = tf.placeholder(tf.int64)

        # Placeholder for observation
        observation = tf.placeholder(tf.float32, shape=(None, obs_dim))
        self.observation = observation

        # Placeholder for actions
        self.actions = tf.placeholder(tf.int64)

        # Placeholder for advantage values
        self.adv_vals = tf.placeholder(tf.float32)

        # Hidden Layer
        W0 = tf.Variable(tf.truncated_normal([obs_dim, hidden_dim],\
                                            stddev=1 / math.sqrt(obs_dim)),\
                                            name="W0")
        b0 = tf.Variable(tf.zeros([hidden_dim]), name="b0")
        h0 = tf.nn.relu(tf.matmul(observation, W0) + b0)

        # Softmax layer
        W1 = tf.Variable(tf.truncated_normal([hidden_dim, act_dim],\
                                            stddev=1 / math.sqrt(hidden_dim)),\
                                            name="W1")
        b1 = tf.Variable(tf.zeros([act_dim]), name="b1")

        # Op for probability distribution over actions
        self.prob = tf.nn.softmax(tf.matmul(h0, W1) + b1)

        # Op for computing loss
        self.loss = tf.matmul(self.prob[tf.range(N), actions], adv_vals) / N

        # Op for computing grads
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(self.loss)

        # Negate the gradients
        new_gradients = []
        for vars, grads in gradients:
            new_gradients.append((vars, -grads))

        # Op for updating params    
        update = optimizer.apply_gradients(new_gradients)

    def take_action(self, obs):
        """
        Perform action by sampling from probability distribution defined by
        policy network over action space
        """
        prob_n = self.prob.eval(feed_dict={self.observation : obs})
        return np.random.choice(self.act_dim, p=prob_n[0])

    def update_policy(self, obs, actions, advantage_vals):
        """
        Computes gradient of the score function estimator wrt to policy params
        and updates params
        """
        self.update.eval(feed_dict = {self.observation : obs,
                                      self.actions : actions,
                                      self.adv_vals : advantage_vals})
