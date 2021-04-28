import numpy as np
from collections import defaultdict
import random


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.00
        self.gamma = 0.9
        self.alpha = 0.06
        self.alpha_decay = 0.9999
        self.alpha_min = 0.01

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # return np.random.choice(self.nA)

        # Choose action according to epsilon greedy policy
        if random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # self.Q[state][action] += 1

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)

        # update the Q table
        current_estimate = self.Q[state][action]
        greedy_estimate = self.Q[next_state][np.argmax(self.Q[next_state])] if next_state is not None else 0
        new_estimate = current_estimate + self.alpha * (reward + self.gamma * greedy_estimate - current_estimate)

        self.Q[state][action] = new_estimate

