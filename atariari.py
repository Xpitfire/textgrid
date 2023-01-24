import gym
from gym import wrappers
from PIL import Image
import numpy as np


env = gym.make('Breakout-ram-v4')
obs, info = env.reset()
print(obs, info)

pixels = env.env.ale.getScreenRGB()
im = Image.fromarray(pixels)
im.save('demo.png')

obs, reward, done, trunc, info = env.step(1)
print(obs, info)

pixels = env.env.ale.getScreenRGB()
im = Image.fromarray(pixels)
im.save('demo1.png')