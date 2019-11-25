import torch
import numpy as np
from scipy.ndimage.measurements import center_of_mass

from model import ReplayMemory, DQN


class TrainPongV0(object):
    """
    Class for training a DQN model for the Pong V0 Network
    """

    def __init__(self, model):
        self.model = model

    def prepare_state(self, s):
        """
        Minimizes memory of state s using two methods:
            - RGB -> Grayscale
            - Cut out unecessary rows
        s is a np.ndarray of shape (210, 160, 3)
        The first 34 rows is bounding bar and score - not useful for training.
        Last 16 rows is bounding bar - also not useful.
        """
        # Get rid of useless rows
        reduced_rows = s[34:194, :, 0]

        # Background is 0, paddles & ball is 1 (R value of backround 144)
        masked = (reduced_rows != 144).astype(int)

        # Center of our paddle (dqn paddle), opponent paddle, and ball y and x coordinates
        dqn_y, _ = center_of_mass(masked[:, 140:144])
        opp_y, _ = center_of_mass(masked[:, 16:20])
        ball_y, ball_x = center_of_mass(masked[:, 20:140])

        dqn_y = 80 if np.isnan(dqn_y) else dqn_y
        opp_y = 80 if np.isnan(opp_y) else opp_y
        ball_x = 80 if np.isnan(ball_x) else ball_x + 20
        ball_y = 80 if np.isnan(ball_y) else ball_y

        return masked, opp_y, dqn_y, ball_x, ball_y
