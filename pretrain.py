#! /usr/bin/env python3

import gym
import time
import numpy as np

from model import ReplayMemory
from train import TrainPongV0


vx = []
vy = []


def get_action(s):
    """Choose an action using a hard coded AI based on current state.

    Args:
        s - Current state. Array with elements 
            [opponent paddle, our paddle, ball x pos, ball y pos, ball x velocity, ball y velocity]
    Returns:
        integer in action space [0,1,2,3,4,5] - represents movements of 
            [do nothing, do nothing, up, down, up, down] 
    """
    global vx, vy

    move = 0
    center_y = 80  # Screen is 160 pixels tall

    ball_x = s[2]
    ball_y = s[3]
    ball_vx = s[4]
    ball_vy = s[5]

    # If ball is moving away from us, center our paddle
    if ball_vx < 0:
        vx = []
        vy = []
        move = s[1] - center_y
        print(f'Move is {move}')


    # If ball is moving towars us, calculate where the y position will be when it reaches our x 
    # position (column 140) anyd move towards it
    if ball_vx > 0:
        print(f'Ball started at: ({ball_x}, {ball_y}) moving with velocity ({ball_vx}, {ball_vy})')

        vx = (vx + [ball_vx])[:5]
        vy = (vy + [ball_vy])[:5]

        ball_vx = sum(vx)/len(vx)
        ball_vy = sum(vy)/len(vy)

        while ball_x < 140:
            ball_x += ball_vx  # Add vx to x
            ball_y += ball_vy  # Add vy to y

            if (ball_y > 159-ball_vy) or (ball_y < -ball_vy):
                ball_vy = -ball_vy
                vy = []

        move = s[1] - ball_y # Move towards predicted y
        print(f'Ball will intersect our paddle at y = {ball_y}')
        print(f'Our paddle is currently at y = {s[1]}')
        print(f'Move is {move}')
        # input()

    # up
    if move > 3:
        return np.random.choice([2,4])

    # down
    if move < -3:
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
                a = get_action(s[0])
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
