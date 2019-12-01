#! /usr/bin/env python3

import gym
import time
import numpy as np

from model import ReplayMemory
from train import TrainPongV0

def get_action():
    # Action Space: integers [0,1,2,3,4,5] - represents movements of [do nothing, do nothing, up, down, up, down]
    key = input()
    if  len(key) == 0:
        return np.random.choice([0,1])
    if len(key) > 1:
        key = key[0]
    # Up
    if key == 'j':
        return np.random.choice([2,4])
    # Down
    if key == 'k':
        return np.random.choice([3,5])
    # Nop
    return np.random.choice([0,1])


def play_game():
    env = gym.make('PongNoFrameskip-v0')
    mem = ReplayMemory(10000)
    try:
        for _ in range(100):
            obs = env.reset()
            s = TrainPongV0.prepare_state(obs)
            for _ in range(10000):
                a = get_action()
                prev_s = s
                obs, r, d, _ = env.step(a)
                s = TrainPongV0.prepare_state(obs, prev_s=prev_s)
                r += TrainPongV0.better_reward(s, scaler=5)
                print(s, end=' ')
                print(r)
                mem.push(prev_s, a, r, s, d)
                env.render()
                if d:
                    break
    except KeyboardInterrupt:
        pass

    np.save('pre_trained_mem', mem, allow_pickle=True)
    env.close()


if __name__ == '__main__':
    play_game()
