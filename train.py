import torch

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
        


