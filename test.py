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

def get_epsilon(steps):
    EPSILON_FINAL = 0.02
    EPSILON_START = 0.3
    EPSILON_DECAY = 10000000
    return EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * np.exp(-1 * steps / EPSILON_DECAY)

def render_model(path, for_gif=False, epsilon=None):
    env = gym.make('PongDeterministic-v4')
    dqn = DQN()
    dqn.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    dqn.eval()

    obs = env.reset()
    s = TrainPongV0.prepare_state(obs)
    epsilon = get_epsilon(5687803)
    print(epsilon)
    frames = []
    tot_reward = 0
    try:
        for i in range(15000):

            if for_gif:
                frames.append(Image.fromarray(env.render(mode='rgb_array')))
            else:
                env.render()

            if np.random.rand() < epsilon:
                a = np.random.choice([1,2,3])
            else:
                with torch.no_grad():
                    a = dqn(torch.from_numpy(s))[0].argmax() + 1

            prev_s = s
            obs, r, d, _ = env.step(a)
            tot_reward += r
            s = TrainPongV0.prepare_state(obs, prev_s=prev_s)
            if d:
                break

    except KeyboardInterrupt:
        pass

    env.close()

    if for_gif:
        return (tot_reward, frames, i)


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


            reward_tuples.append((tot_reward, int(f.replace('HPC_', ''))))

    finally:
        reward_tuples.sort(key=lambda s: s[1])
        ls = [s[1] for s in reward_tuples]
        rs = [s[0] for s in reward_tuples]
        np.save('reward_tuples', reward_tuples)
        plt.plot(ls, rs)
        plt.show()

    env.close()


if __name__ == '__main__':
    # collect_training_rewards('test_model/HPC_Training')
    best_r, best_frames = -1000, []

    r, frames = render_model('policy_episode_14200_8433405', for_gif=False)
    # print([render_model('policy_episode_14200_8433405', for_gif=True)[0] for _ in range(10)])

    # for forle in os.listdir('curr_train'):
    #     for _ in range(5):
    #         r, frames, steps = render_model('curr_train/' + forle, for_gif=True, epsilon=0.1)
    #         print(forle, r)
    #         if r > best_r:
    #             best_r = r
    #             best_frames = frames

    # with open('openai_gym.gif', 'wb') as f:
    #     im = Image.new('RGB', best_frames[0].size)
    #     im.save(f, save_all=True, append_images=best_frames)

