from copy import deepcopy
import itertools
import time

import gym
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from .replay import ReplayBuffer
from .ac import MLPActorCritic

def td3(env_fn, exp_name, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):

    writer = SummaryWriter(f'logs/{exp_name}')
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    test_env = env_fn()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]    # Assumes that all actions have same bound as the first!

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    for p in ac_targ.parameters():
        p.requires_grad = False

    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    replay_buffer = ReplayBuffer(act_dim, obs_dim, replay_size)

    def calc_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)

            epsilon = torch.rand_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        loss_info = {'q1_vals': q1.detach().numpy(), 'q2_vals': q2.detach().numpy()}
        
        return loss_q, loss_info

    def calc_loss_pi(data):
        o = data['obs']
        loss = ac.q1(o, ac.pi(o))
        return -loss.mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data, timer, count):
        q_optimizer.zero_grad()
        loss_q, loss_info = calc_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        if timer % policy_delay == 0:
            for p in q_params:
                p.requires_grad = False 

            pi_optimizer.zero_grad()
            loss_pi = calc_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            writer.add_scalar('loss/avg_pi', loss_pi.item(), count)

            for p in q_params:
                p.requires_grad = True 

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        writer.add_scalar('loss/avg_q1', loss_info['q1_vals'].mean().item(), count)
        writer.add_scalar('loss/avg_q2', loss_info['q2_vals'].mean().item(), count)

    def get_action(obs, noise_scale):
        a = ac.act(torch.as_tensor(obs, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        ep_rewards = []
        ep_lengths = []
        for i in range(num_test_episodes):
            obs = test_env.reset()
            done = False
            ep_ret = 0
            ep_len = 0

            while not (done or (ep_len == max_ep_len)):
                obs, rew, done, _ = test_env.step(get_action(o, 0))
                ep_ret += rew
                ep_len += 1

            ep_rewards.append(ep_ret)
            ep_lengths.append(ep_len)

        return sum(ep_rewards) / len(ep_rewards), sum(ep_lengths) / len(ep_lengths)

    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o = env.reset()
    epoch = 0
    ep_ret = 0
    ep_len = 0
    ep_count = 0
    update_count = 0

    print(f'Running epoch 0....')

    for t in range(total_steps):
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        replay_buffer.store(o, a, o2, r, d)

        o = o2

        if d or (ep_len == max_ep_len):
            writer.add_scalar('episode/reward', ep_ret, ep_count)
            writer.add_scalar('episode/length', ep_len, ep_count)

            o = env.reset()
            ep_ret = 0
            ep_len = 0
            ep_count += 1

        if t >= update_after and t % update_every == 0:
            for i in range(update_every):
                batch = replay_buffer.sample(batch_size)
                update(data=batch, timer=i, count=update_count)
                update_count += 1

        if (t + 1) % steps_per_epoch == 0:

            # save model at end of epoch here, print epoch stuff
            torch.save(ac.state_dict(), f'models/{exp_name}')

            avg_test_reward, avg_test_length = test_agent()
            writer.add_scalar('epoch_test/avg_reward', avg_test_reward, epoch)
            writer.add_scalar('epoch_test/avg_length', avg_test_length, epoch)

            epoch = (t + 1) // steps_per_epoch
            print(f'Running epoch {epoch}...')


    writer.close()
