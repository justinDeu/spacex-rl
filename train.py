import argparse
import os
import shutil
import sys

import gym
import gym.spaces
import rocket_lander_gym

from td3.td3 import td3

BASE_LOG_DIR = 'runs'

parser = argparse.ArgumentParser()
parser.add_argument('exp_name', type=str)
parser.add_argument('--env', type=str, default='RocketLander-v0')
parser.add_argument('--hid', type=int, default=256)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--save-freg', type=int, default=5)
args = parser.parse_args()

exp_path = os.path.join(BASE_LOG_DIR, args.exp_name)

if os.path.exists(exp_path):
    if args.exp_name == 'test' or input(f'Do you want to overwrite experiment {args.exp_name}? [y/n]').lower() == 'y':
        print(f'Deleteing contents of experiment folder: {args.exp_name}')
        shutil.rmtree(exp_path) 
    else:
        print(f'Not overwriting experiment {args.exp_name}. Exiting!')
        sys.exit()

td3(
    lambda : gym.make(args.env),
    args.exp_name, 
    ac_kwargs={'hidden_sizes': [args.hid] * args.l}, 
    gamma=args.gamma,
    seed=args.seed,
    epochs=args.epochs,
    save_freq=args.save_freg
)
