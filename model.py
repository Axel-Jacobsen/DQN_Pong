import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        self.memory.append(Transition(*args))

    # Returns list of Transitions
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def count(self):
        return len(self.memory)


class DQN(nn.Module):
    '''DQN'''

    def __init__(self, in_features=4, n_actions=6, device='cpu'):
        self.device = device

        super(DQN, self).__init__()
        self.gru = nn.GRU(in_features, 32, 2)
        self.linear = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.leaky_relu(self.gru(x.float()))
        return self.linear(x)
