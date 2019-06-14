import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline = T.Compose([T.ToPILImage(),
                    T.Resize([84, 84]),
                    T.ToTensor()])


def state_processor(np_array):
    Y = 0.2116 * np_array[:, :, 0] + 0.7152 * np_array[:, :, 1] + 0.0722 * np_array[:, :, 2]
    return pipeline(torch.from_numpy(np.expand_dims(Y.astype(np.float32), 0))).unsqueeze(0).to(device)


class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(8, 8), stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        conv_out_dim = self._get_conv2d_out_dim(self._get_conv2d_out_dim(self._get_conv2d_out_dim(h, 8, 4), 4, 2), 3, 1)
        self.fc1 = nn.Linear(in_features=conv_out_dim**2*64, out_features=512)
        self.bn4 = nn.BatchNorm1d(512)
        self.output = nn.Linear(in_features=512, out_features=4)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.fc1((x.view(x.size(0), -1)))))
        return self.output(x)

    def _get_conv2d_out_dim(self, input_dim, kernel_size, stride, pad=0):
        out_dim = np.floor((input_dim - kernel_size + 2 * pad) / stride + 1)
        return int(out_dim)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
