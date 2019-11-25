#!/usr/bin/env python3

import gym
import numpy as np
import time
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from train import TrainPongV0


def convert_gray2rgb(image):
    width, height = image.shape
    out = np.empty((width, height, 3), dtype=np.uint8)
    out[:, :, 0] = image * 255
    out[:, :, 1] = image * 255
    out[:, :, 2] = image * 255
    return out


def render_model():
    trainer = TrainPongV0(None)
    env = gym.make('Pong-v0')

    dt = []
    s = env.reset()
    try:
        for _ in range(1000):

            a = env.action_space.sample()
            s, _, d, _ = env.step(a)

            t1 = time.time()
            sn, opp_y, dqn_y, ball_x, ball_y = trainer.prepare_state(s)
            dt.append(time.time() - t1)
            sprime = convert_gray2rgb(sn)

            img = Image.fromarray(sprime, 'RGB')
            img.putpixel((int(ball_x), int(ball_y)), (255, 0, 0))
            img.putpixel((19, int(opp_y)), (255, 0, 0))
            img.putpixel((140, int(dqn_y)), (255, 0, 0))
            img.show()

            if d:
                break
    except KeyboardInterrupt:
        pass

    plt.plot(dt)
    plt.show()
    env.close()


if __name__ == '__main__':

    render_model()
