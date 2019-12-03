#! /usr/bin/env python3

import os
import sys
import gym
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from model import DQN
from train import TrainPongV0


def render_model(path):
    env = gym.make('Pong-v4')
    dqn = DQN()
    dqn.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    dqn.eval()

    obs = env.reset()
    s = TrainPongV0.prepare_state(obs)
    epsilon = 0.05
    try:
        for _ in range(15000):
            env.render()
            if np.random.rand() < epsilon:
                a = np.random.choice(range(0,6))
            else:
                a = dqn(s).argmax()
            prev_s = s
            obs, _, d, _ = env.step(a)
            s = TrainPongV0.prepare_state(obs, prev_s=prev_s)
            if d:
                break

    except KeyboardInterrupt:
        pass

    env.close()


def collect_training_rewards(policy_dir_path):
    env = gym.make('Pong-v4')
    epsilon = 0.05
    reward_tuples = []

    try:
        for f in os.listdir(policy_dir_path):
            print(f'processing {f}')
            dqn = DQN()
            dqn.load_state_dict(
                    torch.load(policy_dir_path + '/' + f, map_location=torch.device('cpu')))
            dqn.eval()

            obs = env.reset()
            s = TrainPongV0.prepare_state(obs)
            tot_reward = 0

            while True:
                if np.random.rand() < epsilon:
                    a = np.random.choice(range(0,6))
                else:
                    a = dqn(s).argmax()

                prev_s = s
                obs, r, d, _ = env.step(a)
                s = TrainPongV0.prepare_state(obs, prev_s=prev_s)

                tot_reward += r

                if d:
                    break

            reward_tuples.append((tot_reward, int(f.replace('policy_episode_', ''))))

    finally:
        reward_tuples.sort(key=lambda s: s[1])
        ls = [s[1] for s in reward_tuples]
        rs = [s[0] for s in reward_tuples]
        np.save('reward_tuples', reward_tuples)
        plt.plot(ls, rs)
        plt.show()

    env.close()


if __name__ == '__main__':
    collect_training_rewards('test_model/desperate_policies')

