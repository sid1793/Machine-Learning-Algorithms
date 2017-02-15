import numpy as np
from collections import defaultdict

def run_episode(agent, env, max_episode_length=25, render=False):
    """
    Runs an episode for the environment and returns dictionary of observations,
    rewards and actions
    """
    obs = env.reset()
    observations = [obs]
    rewards = []
    actions = []
    for i in range(max_episode_length):
        obs = obs.reshape(1,env.observation_space.shape[0]) 
        a = agent.take_action(obs)
        obs, rew, done, _ = env.step(a)
        observations.append(obs)
        rewards.append(rew)
        actions.append(a)
        if done: break
        if render: env.render()
    return {"observations": observations[:-1],
            "rewards": rewards,
            "actions": actions
           }

def discounted_returns(rewards, gamma = 0.9):
    """
    Returns a list of discounted rewards for an episode at each time step
    discounted exponentially by factor gamma
    """
    returns = [0]
    for idx, rew in enumerate(reversed(rewards)):
        disc_ret = rew + returns[idx] * gamma
        returns.append(disc_ret)
    returns.reverse()
    return returns[:-1]

def compute_baseline(observations, returns):
    """
    Returns a dictionary of the average returns from each state that serves as
    the baseline
    """
    # TO-DO - Implement approximator of value function
    raise NotImplementedError
