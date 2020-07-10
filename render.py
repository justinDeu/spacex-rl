import torch
import numpy as np
import gym
import gym.spaces
import rocket_lander_gym
from gym import wrappers

from td3.ac import MLPActorCritic

def render(model_file, output, env, force=False):

    env_to_wrap = gym.make(env)
    env = wrappers.Monitor(env_to_wrap, output, force=force)

    ac = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=[512, 512])
    ac.load_state_dict(torch.load(model_file))

    observation = env.reset()
    while True:
        action = ac.act(torch.as_tensor(observation, dtype=torch.float32))

        observation, reward, done, info = env.step(action)

        print("Action Taken  ", action)
        print("Observation   ", observation)
        print("Reward Gained ", reward)
        print("Done          ", done)
        print("Info          ", info, end='\n\n')

        if done:
            print("Simulation done.")
            break

    env.close()
    env_to_wrap.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--env', type=str, default='RocketLander-v0')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    render(args.model, args.output, args.env, force=args.force)

