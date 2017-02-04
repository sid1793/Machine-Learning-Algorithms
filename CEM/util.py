import numpy as np
import policy
from gym.spaces import Discrete, Box

def make_policy(env, theta):
    if isinstance(env.action_space, Discrete):
        return policy.LinearDetDiscActSpace(theta, env.observation_space, env.action_space)
    elif isinstance(env.action_space, Box):
        return policy.LinearDetContActSpace(theta, env.observation_space, env.action_space)
    else:
       raise NotImplementedError

def run_episode(env, theta, num_steps, render = False):
    pol = make_policy(env, theta)
    obs = env.reset()
    total_rew = 0
    for t in range(num_steps):
        action = pol.take_action(obs)
        obs, reward, done, _ = env.step(action) 
        total_rew += reward
        if render: env.render()
        if done: break
    return total_rew
