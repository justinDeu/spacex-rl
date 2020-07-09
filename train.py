import argparse

import gym
import gym.spaces
import rocket_lander_gym

from td3.td3 import td3



parser = argparse.ArgumentParser()
parser.add_argument('exp_name', type=str)
parser.add_argument('--env', type=str, default='RocketLander-v0')
parser.add_argument('--hid', type=int, default=256)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

td3(
    lambda : gym.make(args.env),
    args.exp_name, 
    ac_kwargs={'hidden_sizes': [args.hid] * args.l}, 
    gamma=args.gamma,
    seed=args.seed,
    epochs=args.epochs,
)
