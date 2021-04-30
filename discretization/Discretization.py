import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display

# Set plotting options
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

display = Display(visible=0, size=(1400, 900))
display.start()

is_ipython = 'inline' in plt.get_backend()
plt.ion()

# Create an environment and set random seed
env = gym.make('MountainCar-v0')
env.seed(505);

state = env.reset()
img = plt.imshow(env.render(mode='rgb_array'))
for t in range(1000):
    action = env.action_space.sample()
    img.set_data(env.render(mode='rgb_array'))
    plt.axis('off')
    display.display(plt.gcf())
    display.clear_output(wait=True)
    state, reward, done, _ = env.step(action)
    if done:
        print('Score: ', t + 1)
        break

env.close()
