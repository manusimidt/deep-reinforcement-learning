import sys
import time

import gym
import numpy as np
from collections import defaultdict

from gym.envs.toy_text import BlackjackEnv

from plot_utils import plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v0')


def generate_episode_from_Q(env: BlackjackEnv, Q, epsilon, action_count):
    """
    Generates an episode
    @param env:
    @param Q:
    @param epsilon:
    @param action_count
    """
    episode = []
    state = env.reset()
    while True:
        action = np.random.choice(np.arange(action_count), p=get_probs(Q[state], epsilon, action_count)) \
            if state in Q else env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s


def update_Q(env, episode, Q, alpha, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]]
        Q[state][actions[i]] = old_Q + alpha * (sum(rewards[i:] * discounts[:-(1 + i)]) - old_Q)
    return Q


def mc_control(env: BlackjackEnv, num_episodes: int, alpha: float, gamma: float = 1.0, eps_start=1.0, eps_decay=.9999,
               eps_min=0.05) -> tuple:
    """
    Every visit monte carlo control
    @param env: Blackjack Environment from gym
    @param num_episodes:
    @param alpha:
    @param gamma:
    @param eps_start: Initialize Value for epsilon
    @param eps_decay: Amount of decay per episode for epsilon current_epsilon = (initial_epsilon)^#episode
    @param eps_min: Minimum number of epsilon, makes sure that the agent always has some type of exploration
    """
    action_count = env.action_space.n
    # initialize empty dictionary of arrays, prevents KeyError, returns np.zeros(nA) instead
    Q = defaultdict(lambda: np.zeros(action_count))
    epsilon = eps_start
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # set the value of epsilon
        epsilon = max(epsilon * eps_decay, eps_min)
        # generate an episode by following epsilon-greedy policy
        episode = generate_episode_from_Q(env, Q, epsilon, action_count)
        # update the action-value function estimate using the episode
        Q = update_Q(env, episode, Q, alpha, gamma)
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy, Q


policy, Q = mc_control(env, 500000, 0.02)
V = dict((k, np.max(v)) for k, v in Q.items())
plot_blackjack_values(V)
plot_policy(policy)
