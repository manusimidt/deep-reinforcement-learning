import sys
import gym
import numpy as np
from collections import defaultdict

from plot_utils import plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v0')
print(env.observation_space)
print(env.action_space)

for i_episode in range(300):
    state = env.reset()
    while True:
        print(f"Player sum: {state[0]}\tDealer's card: {state[1]}\tPlayer has ace:{state[2]}")
        action = env.action_space.sample()
        print(f"Player decides to {'HIT' if action == 0 else 'STICK'}")
        state, reward, done, info = env.step(action)
        if done:
            print('You won :)\n') if reward > 0 else print('You lost :(\n')
            break
