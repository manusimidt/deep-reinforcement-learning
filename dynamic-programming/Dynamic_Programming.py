from frozenlake import FrozenLakeEnv

env = FrozenLakeEnv()

# print the state space and action space
print(env.observation_space)
print(env.action_space)

# print the total number of states and actions
print(env.nS)
print(env.nA)

# [(probability, next_state, reward, done),...]
# where prop is P(reward|next_state) and done is true if next_state=terminal_state
print(env.P[1][0])
