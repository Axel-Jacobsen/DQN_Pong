import random
from collections import namedtuple, deque

import torch.nn.functional as F
import torch.nn as nn


class ReplayMemory(object):
    Transition = namedtuple('Transition',
                            ('state', 'action', 'reward', 'next_state', 'done'))

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def count(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=6):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linr1 = nn.Linear(7 * 7 * 64, 512)
        self.linr2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linr1(x.view(x.size(0), -1)))
        return self.linr2(x)
