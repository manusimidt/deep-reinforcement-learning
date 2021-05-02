import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from gym.envs.toy_text import CliffWalkingEnv
import random
import check_test
from plot_utils import plot_values

env: CliffWalkingEnv = gym.make('CliffWalking-v0')


def expected_sarsa(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.9999, eps_min=0.05) -> tuple:
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))

    step_count_storage = []
    epsilon = eps_start
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}/{num_episodes}, Current average step count {np.average(step_count_storage)}")
        step_count = 0
        epsilon = 0.005
        # set the initial state
        state = env.reset()

        # loop over the steps in the episode
        while True:
            action = choose_action_epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, info = env.step(action)
            step_count += 1

            # expected sarsa uses ...
            # update the Q table now
            current_estimate = Q[state][action]
            prop: tuple = get_probabilities(Q, next_state, epsilon)
            expected_return = sum([prop[a] * Q[next_state][a] for a in np.arange(env.nA)])
            new_estimate = current_estimate + alpha * (reward + gamma * expected_return - current_estimate)
            Q[state][action] = new_estimate

            if not done:
                state = next_state
            else:
                break
        step_count_storage.append(step_count)
    return Q


def choose_action_epsilon_greedy(Q, state, epsilon) -> int:
    """
    Returns a action based on the epsilon greedy policy
    """
    # if state not in Q:
    #     action = env.action_space.sample()
    # else:
    #     # choose the action according to the policy
    #     probabilities: tuple = get_probabilities(Q, state, epsilon)
    #     action = np.random.choice(np.arange(env.nA), p=probabilities)
    # return action
    if random.random() > epsilon:  # select greedy action with probability epsilon
        return np.argmax(Q[state])
    else:  # otherwise, select an action randomly
        return random.choice(np.arange(env.action_space.n))


def get_probabilities(Q, state, epsilon) -> tuple:
    """
    @param Q: The Q-Table
    @param state: the current state
    @param epsilon:
    returns an array of probabilities for each action
    """
    possible_actions_count = len(Q[state])
    policy_s: [] = np.ones(possible_actions_count) * epsilon / possible_actions_count
    # look what is the best action according to the Q-Table
    best_a: int = np.argmax(Q[state])
    policy_s[best_a] = 1 - epsilon + (epsilon / possible_actions_count)
    return policy_s


# obtain the estimated optimal policy and corresponding action-value function
Q_expsarsa = expected_sarsa(env, 5000, 1)

# print the estimated optimal policy
policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_expsarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_expsarsa)

# plot the estimated optimal state-value function
plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])