#! /usr/bin/env python3

import sys
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch

from PIL import Image, ImageDraw

from model import DQN
from train import TrainPongV0

def render_model(path):
    env = gym.make('Pong-v0')
    dqn = DQN()
    dqn.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    dqn.eval()

    dt = []
    obs = env.reset()
    s = TrainPongV0.prepare_state(obs)
    try:
        for _ in range(1000):
            env.render()
            a = dqn(s).argmax()
            prev_s = s
            obs, _, d, _ = env.step(a)
            s = TrainPongV0.prepare_state(obs, prev_s=prev_s)

            if d:
                break
    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            print(path)
            render_model(path)
    else:
        path = 'HPC_1'
        render_model(path)

