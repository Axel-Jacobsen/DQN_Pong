#! /usr/bin/env python3

import gym
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy.ndimage.measurements import center_of_mass
import random

from model import ReplayMemory, DQN, Transition


class TrainPongV0(object):
    """
    Class for training a DQN model for the Pong V0 Network
    """

    # hyperparameters
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPSILON_START = 1
    EPSILON_FINAL = 0.05
    EPSILON_DECAY = 10000000
    TARGET_UPDATE = 100
    lr = 1e-5
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 5 * INITIAL_MEMORY

    def __init__(self, target: DQN, policy: DQN, memory: ReplayMemory, device):
        self.target = target
        self.policy = policy
        self.memory = memory
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.steps = 0
        self.episodes = 0
        self.device = device
        self.total_rewards = []  # Total rewards - dt is 1 episode

    @property
    def epsilon(self):
        return (self.EPSILON_FINAL + (self.EPSILON_START - self.EPSILON_FINAL) * np.exp(-1 * self.steps / self.EPSILON_DECAY))

    @staticmethod
    def prepare_state(s: np.ndarray, prev_s=None):
        """Returns the simplified state for the last 5 timesteps.

        Shape of the simplified state is (time_seq, batch, input_size)

        Args:
            s (ndarray) - Game state returned by Gym for the current time step
            prev_s (ndarray) - Previous state in the format returned by this
                function. Shape is (time_seq, batch, input_size)
        """
        if prev_s is None:
            prev_s = np.zeros((10, 1, 4))

        # Get rid of useless rows and the green & blue colour chanels
        reduced_rows = s[34:194, :, 0]

        # Background is 0, paddles & ball is 1 (R value of backround 144)
        masked = (reduced_rows != 144).astype(int)

        # Center of our paddle (dqn paddle), opponent paddle, and ball y and x coordinates
        dqn_y, _ = center_of_mass(masked[:, 140:144])
        opp_y, _ = center_of_mass(masked[:, 16:20])
        ball_y, ball_x = center_of_mass(masked[:, 20:140])

        dqn_y = 80 if np.isnan(dqn_y) else dqn_y
        opp_y = 80 if np.isnan(opp_y) else opp_y
        # x position of ball is offset by 21 px to the center of img
        ball_x = 80 if np.isnan(ball_x) else ball_x + 21
        ball_y = 80 if np.isnan(ball_y) else ball_y

        state_vec = np.array([[[opp_y, dqn_y, ball_x, ball_y]]])
        state_vec = np.concatenate((state_vec, prev_s))[:10]

        return state_vec

    def select_action(self, state, env):
        """Select an action using randomized greedy.

        action space: integers [0,1,2,3,4,5] - represents movements of 
        [do nothing, do nothing, up, down, up, down]

        In our networks action space it will output a number between 0 and 2.
        0 is NOP, 1 is up, 2 is down. Adding 1 to this will give correct mapping to
        actions as defined by the environment

        If a random number is below epsilon, sample a random action from the action 
        space. Otherwise, choose the best action according to the current policy.

        Args:
            state (ndarray) - Array of shape (time_seq, batch, input_size) consisting of 
                paddle and ball positions for the last 5 frames.
            env - Gym environment
        """
        if np.random.rand() < self.epsilon:
            return torch.tensor(random.choice([0, 1, 2]), device=self.device)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state)
                res = self.policy(state.to(self.device))
                return res[0].argmax()  # Max from the most recent time step

    def memory_replay(self):
        """
        This method was more or less copied from https://github.com/jmichaux/dqn-pytorch/blob/master/main.py#L38 - It is a very clean solution, very readable. 
        """
        if len(self.memory) < self.BATCH_SIZE:
            return

        # Returns list of Transitions
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        actions = tuple(
            (map(lambda a: torch.tensor([[a]], device=self.device), batch.action)))

        rewards = tuple(
            (map(lambda r: torch.tensor([r], device=self.device), batch.reward)))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device)

        non_final_next_states = torch.tensor([s for s in batch.next_state
                                              if s is not None]).to(self.device)
        non_final_next_states = non_final_next_states.squeeze().transpose(0, 1)

        unwrapped_states = np.array(batch.state).squeeze()
        # Reshape from (batch, time, input) to (time, batch, input)
        unwrapped_states = np.transpose(unwrapped_states, (1, 0, 2))

        state_batch = torch.tensor(unwrapped_states).to(self.device)
        action_batch = torch.tensor(actions).to(self.device)
        reward_batch = torch.tensor(rewards).to(self.device)

        # Value of current state as predicted by policy network
        # index 0 gets most recent timestep
        state_action_values = self.policy(state_batch)[0]
        state_action_values = state_action_values.gather(
            1, action_batch.reshape((-1, 1)))

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target(
            non_final_next_states)[0].max(1)[0].detach()

        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        a = list(self.policy.parameters())[0].clone()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        b = list(self.policy.parameters())[0].clone()
        assert not torch.equal(a.data, b.data)

    @staticmethod
    def load_memory(path):
        return (np.load(path, allow_pickle=True)).item()

    def train(self, num_episodes: int):
        env = gym.make('PongDeterministic-v4')

        batch_reward = 0.0

        for episode in range(num_episodes):
            self.episodes += 1
            state = env.reset()
            state = self.prepare_state(state)
            tot_reward = 0

            while True:
                action = self.select_action(state, env)
                # Actions are range [0,2] but env expects [1,3]
                obs, reward, done, _ = env.step(action+1)
                self.steps += 1

                if not done:
                    next_state = self.prepare_state(obs, prev_s=state)
                else:
                    next_state = None

                tot_reward += reward
                batch_reward += reward
                reward = torch.tensor([reward], device=self.device)

                self.memory.push(state, action.to(self.device),
                                 reward.to(self.device), next_state, done)
                state = next_state

                if self.steps > self.INITIAL_MEMORY or len(self.memory) >= self.INITIAL_MEMORY:
                    self.memory_replay()
                    if self.steps % self.TARGET_UPDATE == 0:
                        self.target.load_state_dict(policy.state_dict())

                if done:
                    break

            self.total_rewards.append(tot_reward)

            if (self.episodes) % 20 == 0:
                print('\rTotal steps: {} \t Episode: {}/{} \t Batch reward: {:.3f} \t Last reward: {:.3f} \t Epsilon: {:.3f}'.format(
                    self.steps, episode+1, num_episodes, batch_reward, tot_reward, self.epsilon))

                batch_reward = 0

                if (self.episodes) % 100 == 0:
                    policy_PATH = f'policies/policy_episode_{self.episodes}_{self.steps}'
                    target_PATH = f'targets/target_episode_{self.episodes}_{self.steps}'
                    torch.save(self.policy.state_dict(), policy_PATH)
                    torch.save(self.target.state_dict(), target_PATH)

        policy_PATH = f'policy_episode_{self.episodes}_{self.steps}'
        target_PATH = f'target_episode_{self.episodes}_{self.steps}'
        torch.save(self.policy.state_dict(), policy_PATH)
        torch.save(self.target.state_dict(), target_PATH)

        env.close()


"""UNCOMMENT IF YOU WANT TO TRAIN"""
# if __name__ == '__main__':
#     device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu"
#             )

#     print(f'Using Device {device}')

#     target = DQN(device=device).to(device)
#     policy = DQN(device=device).to(device)
#     mem = ReplayMemory(TrainPongV0.MEMORY_SIZE)
#     trainer = TrainPongV0(target, policy, mem, device)

#     try:
#         trainer.train(50000)
#     finally:
#         np.save('rewards', trainer.total_rewards, allow_pickle=True)
