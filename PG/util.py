import numpy as np

def run_episode(agent, env, max_episode_length, render = False):
    """
    Runs an episode for the environment and returns dictionary of observations,
    rewards and actions
    """
    obs = env.reset()
    observations = [obs]
    rewards = []
    actions = []
    for i in range(max_episode_length):
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
