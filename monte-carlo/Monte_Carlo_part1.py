import sys
import time

import gym
import numpy as np
from collections import defaultdict

from gym.envs.toy_text import BlackjackEnv

from plot_utils import plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v0')


def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def print_episode(episode: tuple) -> None:
    """
    Prints a single episode so that i can understand the tuple contents...
    """
    for x in range(len(episode)):
        print(
            f'S{x}={episode[x][0]} (player score, dealer cards, ace?), A{x}={episode[x][1]} ({"HIT" if episode[x][1] == 0 else "STICK"}), R={episode[x][2]}')


def mc_prediction_q(env: BlackjackEnv, num_episodes: int, generate_episode, gamma: float = 1.0):
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        ## TODO: complete the function
        episode: tuple = generate_episode(env)
        states = [x[0] for x in episode]
        actions = [x[1] for x in episode]
        rewards = [x[2] for x in episode]
        discounted_rewards = [rewards[x] * gamma**x for x in range(len(rewards))]
        for i, state in enumerate(states):
            returns_sum[state][actions[i]] += sum(discounted_rewards)
            N[state][actions[i]] += 1.0
            Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]
    return Q


# obtain the action-value function
Q = mc_prediction_q(env, 500000, generate_episode_from_limit_stochastic)

# obtain the corresponding state-value function
V_to_plot = dict((k, (k[0] > 18) * (np.dot([0.8, 0.2], v)) + (k[0] <= 18) * (np.dot([0.2, 0.8], v))) \
                 for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V_to_plot)
