import numpy as np
import gym
from gym.spaces import Discrete, Box
from util import run_episode

def cem(env, num_steps = 500, n_iter = 100, batch_size = 25, elite_frac = 0.2):
 
    if isinstance(env.action_space, Discrete):
        theta_dim = (env.observation_space.shape[0] + 1) * env.action_space.n
    elif isinstance(env.action_space, Box):
        theta_dim = (env.observation_space.shape[0] + 1) * env.action_space.shape[0]
    else:
        raise NotImplementedError

    theta_mean = np.zeros(theta_dim)
    theta_std = np.ones(theta_dim)

    for i in range(n_iter):
        thetas = [np.random.multivariate_normal(theta_mean, np.diagflat(theta_std)) for j in range(batch_size)]
        rewards = [run_episode(env, theta, num_steps) for theta in thetas] 
        num_elite = int(elite_frac * batch_size)
        elite_idxs = np.argsort(rewards)[batch_size - num_elite:]
        elite_thetas = np.array([thetas[idx] for idx in elite_idxs])
        theta_mean = np.mean(elite_thetas, axis = 0)
        theta_std = np.mean(np.square(elite_thetas - theta_mean.reshape(1, theta_dim)), axis = 0)
        print 'Iteration {}, Mean Reward = {}, Max Reward = {}'.format(i, np.mean(rewards), np.max(rewards))
        run_episode(env, theta_mean, 1000, render = True)

def main():
    env = gym.make('CartPole-v0')
    cem(env)

if __name__ == "__main__":
    main()
