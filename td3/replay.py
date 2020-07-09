import numpy as np
import torch

def join_shapes(size, shape=None):
    if not shape:
        return (size,)
    return (size, shape) if np.isscalar(shape) else (size, *shape)

class ReplayBuffer:
    ''' A FIFO buffer to store and sample state transitions for an RL agent'''

    def __init__(self, act_dim, obs_dim, size):
        self.act_buffer = np.zeros(join_shapes(size, act_dim), dtype=np.float32)
        self.obs_buffer = np.zeros(join_shapes(size, obs_dim), dtype=np.float32)
        self.obs2_buffer = np.zeros(join_shapes(size, obs_dim), dtype=np.float32)
        self.rew_buffer = np.zeros(size, dtype=np.float32)
        self.done_buffer = np.zeros(size, dtype=np.float32)

        self.loc = 0
        self.size = 0
        self.max_size = size


    def store(self, obs, action, obs2, reward, done):
        self.obs_buffer[self.loc] = obs
        self.act_buffer[self.loc] = action
        self.obs2_buffer[self.loc] = obs2
        self.rew_buffer[self.loc] = reward
        self.done_buffer[self.loc] = done

        self.loc = (self.loc + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size=32):
        ids = np.random.randint(0, self.size, size=batch_size)
        batch = {
            'obs': self.obs_buffer[ids],
            'act': self.act_buffer[ids],
            'obs2': self.obs2_buffer[ids],
            'rew': self.rew_buffer[ids],
            'done': self.done_buffer[ids]
        }
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
