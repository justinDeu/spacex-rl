import gym
import gym.spaces
import rocket_lander_gym

from td3.td3 import td3

env_fn = lambda: gym.make('RocketLander-v0')

td3(env_fn, 'test')
