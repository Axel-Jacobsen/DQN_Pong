#! /usr/bin/env python3

import gym
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy.ndimage.measurements import center_of_mass

from model import ReplayMemory, DQN, Transition
from wrappers import wrap_dqn


class TrainPongV0(object):
    """
    Class for training a DQN model for the Pong V0 Network
    """

    # hyperparameters
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPSILON_FINAL = 0.05
    EPSILON_DECAY = 1e6
    TARGET_UPDATE = 1000
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY

    def __init__(self, target: DQN, policy: DQN, memory: ReplayMemory, device):
        self.target = target
        self.policy = policy
        self.memory = memory
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.steps = 0
        self.device = device

    @staticmethod
    def prepare_state(s: np.ndarray, prev_s=None, view=False):
        """
        Boils state down to just the important info:
            - where is the opponent paddle?
            - where is our paddle?
            - where is the ball?
            - where is the ball going?
        s is a np.ndarray of shape (210, 160, 3)
        The first 34 rows is bounding bar and score - not useful for training.
        Last 16 rows is bounding bar - also not useful.
        Colour definitions are in the README - the important piece of info is that the background's R value (of RGB) is 144 - therefore we mask for that value
        returns a torch.tensor with (opponent paddle, our paddle, ball x pos, ball y pos, ball x velocity, ball y velocity)
        """
        if prev_s is None:
            prev_s = torch.from_numpy(np.array([80, 80, 80, 80])).unsqueeze(0)

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

        # velocity is ds/dt
        # without the if/else keeping the magnitude of velocity below 5, the velocity
        # can get really high once someone scores and the ball is placed in the center.
        # the velocities can be 2,3,4 (Pong-v0 randomly skips 2,3,4 frames)
        ball_vx = (ball_x - prev_s[0][2]) if abs(ball_x - prev_s[0][2]) < 5 else 0
        ball_vy = (ball_y - prev_s[0][3]) if abs(ball_y - prev_s[0][3]) < 5 else 0

        # Scale the positions to [0, 10] and leave velocities in [0,4]
        # Hypothesis: Before, we were scaling an int in [0,160] to a float in [0,1]
        # Maybe this range is too small?
        state_vec = np.array([opp_y, dqn_y, ball_x, ball_y, ball_vx, ball_vy]) # / np.array([1, 1, 1, 1, 1, 1])
        torch_state_vec = torch.from_numpy(state_vec).unsqueeze(0)

        if view:
            return torch_state_vec, masked
        else:
            return torch_state_vec

    def select_action(self, state, env):
        # Action Space: integers [0,1,2,3,4,5] - represents movements of [do nothing, do nothing, up, down, up, down]
        # This is redundant; we want our network output to be smaller, cause then it will be easier to train.
        # Therefore, class DQN (in model.py) has 3 outputs
        #   - if the zeroth is biggest, NOP
        #   - if the first is biggest, up
        #   - if the third is biggest, down
        # conviniently, if we add 1 to our network output, we get the gym env's action space for the correct mapping
        if np.random.rand() < (self.EPSILON_FINAL + (1 - self.EPSILON_FINAL) * np.exp(-1 * self.steps / self.EPSILON_DECAY)):
            return torch.tensor(env.action_space.sample(), device=self.device)
        else:
            with torch.no_grad():
                return self.policy(state.to(self.device)).argmax()

    def memory_replay(self):
        if len(memory) < self.BATCH_SIZE:
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

        non_final_next_states = torch.cat([s for s in batch.next_state
            if s is not None]).to(self.device)

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        state_action_values = self.policy(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target(
                non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (
                next_state_values * self.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values,
                expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        a = list(self.policy.parameters())[0].clone()
        loss.backward()
#        for param in self.policy.parameters():
#            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        b = list(self.policy.parameters())[0].clone()
        assert not torch.equal(a.data,b.data)


    def better_reward(self, s):
        """Returns a small reward for the paddle being close to the ball, when the ball
        """
        dqn_y  = s[0][1]
        ball_x = s[0][2]
        ball_y = s[0][3]
        if ball_x > 130 and s is not None:
            return max(0, 24 - abs(ball_y - dqn_y)) / 100
        return 0


    def train(self, num_episodes: int):
        env = gym.make('PongNoFrameskip-v0')
        env = wrap_dqn(env)

        for episode in range(num_episodes):
            state = env.reset()
            state = self.prepare_state(state)
            tot_reward = 0
            while True:

                action = self.select_action(state, env)
                obs, reward, done, _ = env.step(action)
                self.steps += 1

                if not done:
                    next_state = self.prepare_state(obs, prev_s=state)
                else:
                    next_state = None
 
                reward += self.better_reward(next_state)
                tot_reward += reward
                reward = torch.tensor([reward], device=self.device)

                self.memory.push(state, action.to(self.device),
                        reward.to(self.device), next_state, done)
                state = next_state

                if self.steps > self.INITIAL_MEMORY:
                    self.memory_replay()

                    # if self.steps % self.TARGET_UPDATE == 0:
                    # If you uncomment the above line, right indend the below line
                    self.target.load_state_dict(policy.state_dict())

                if done:
                    break

            if episode % 20 == 0:
                print('\rTotal steps: {} \t Episode: {}/{} \t Total reward: {} \t Epsilon: {:.3f}'.format(
                    self.steps, episode, num_episodes, tot_reward, self.EPSILON_FINAL + (1 - self.EPSILON_FINAL) * np.exp(-1 * self.steps / self.EPSILON_DECAY)))

                if episode % 100 == 0:
                    policy_PATH = f'policies/policy_episode_{episode}'
                    target_PATH = f'targets/target_episode_{episode}'
                    torch.save(self.policy.state_dict(), policy_PATH)
                    torch.save(self.target.state_dict(), target_PATH)


        policy_PATH = f'policy_episode_{episode}'
        target_PATH = f'target_episode_{episode}'
        torch.save(self.policy.state_dict(), policy_PATH)
        torch.save(self.target.state_dict(), target_PATH)

        env.close()


if __name__ == '__main__':
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            )
    
    print(f'Using Device {device}')

    target = DQN(device=device).to(device)
    policy = DQN(device=device).to(device)
    target.load_state_dict(policy.state_dict())

    memory = ReplayMemory(TrainPongV0.MEMORY_SIZE)

    trainer = TrainPongV0(target, policy, memory, device)

    trainer.train(4000)
