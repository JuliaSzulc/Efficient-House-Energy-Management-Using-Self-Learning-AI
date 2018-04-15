"""This module provides the clue of the project - RL agent.

It works with environment by taking actions and gaining observations and
reward, and its objective is to maximalize cost function in a continuous
environment.

"""
import random
from collections import deque
import numpy as np
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
import torch.nn.functional as F


class Net(torch.nn.Module):
    """Neural Network with variable layer sizes and 2 hidden layers."""

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden1_size)
        self.fc2 = torch.nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = torch.nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Memory(deque):
    """Subclass of deque, storing transitions batches for Agent"""
    # TODO: Prioritized Experience Replay

    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)


class Agent:
    """Reinforcement Learning agent.

    Agent interacts with the environment, gathering
    information about the reward for his previous actions,
    and observation of state transitions.

    """

    def __init__(self, env):
        self.env = env
        self.actions = None
        self.q_network = None
        self.target_network = None  # this one has "fixed" weights
        self.initial_state = None
        self.current_state = None
        self.memory = Memory(maxlen=5000)
        self.gamma = 0
        self.epsilon = 0
        self.epsilon_decay = 0
        self.epsilon_min = 0
        self.batch_size = 0
        self.l_rate = 0
        self.optimizer = None

        self.reset()

    def reset(self):
        """Initialize the networks and other parameters"""
        self.initial_state = self.env.reset()
        self.actions = self.env.get_actions()

        input_size = len(self.initial_state)
        hidden1_size = 50
        hidden2_size = 20
        output_size = len(self.actions)

        self.q_network = Net(
            input_size,
            hidden1_size,
            hidden2_size,
            output_size
        )
        self.target_network = Net(
            input_size,
            hidden1_size,
            hidden2_size,
            output_size
        )
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1
        self.batch_size = 16
        self.l_rate = 0.01
        self.optimizer = optim.Adagrad(self.q_network.parameters(),
                                       lr=self.l_rate)

    def run(self):
        """Main agent's function. Performs the deep q-learning algorithm"""
        self.current_state = self.env.reset()
        total_reward = 0
        terminal_state = False
        while not terminal_state:
            action_index = self._get_next_action()
            next_state, reward, terminal_state = \
                self.env.step(self.actions[action_index])

            # clip the reward
            if reward < -2:
                reward = -2

            self.memory.append((self.current_state, action_index, reward,
                                next_state, terminal_state))

            # print("Reward = ", reward)
            self.current_state = next_state
            total_reward += reward
            self._train()

            # Update the target network:
            qt = 0.2  # q to target ratio
            for target_param, q_param in zip(self.target_network.parameters(),
                                             self.q_network.parameters()):
                target_param.data.copy_(q_param.data * qt
                                        + target_param.data * (1.0 - qt))

        self.epsilon_min *= self.epsilon_decay
        self.epsilon_min = max(0.01, self.epsilon_min)

        return total_reward

    def _train(self):
        """Trains the underlying q_network with use of experience memory

        Note: this method does a training *step*, not whole training

        """
        # TODO make notebook (and maintain it) explaining this function
        # TODO then move the current comments there
        if len(self.memory) > self.batch_size:
            # Sample random transition batch and transform it into separate
            # batches (of autograd.Variable type)
            exp_batch = self.get_experience_batch()

            input_state_batch = exp_batch[0]
            action_batch = exp_batch[1]
            reward_batch = exp_batch[2]
            next_state_batch = exp_batch[3]
            terminal_mask_batch = exp_batch[4]

            # As q learning states, we want to calculate the error:
            # Q(s,a) - (r + max{a}{Q(s_next,a)})

            # 1. Calculate Q(s,a) for each input state
            all_q_values = self.q_network(input_state_batch)

            # 2. Retrieve q_values only for actions that were taken
            # This use of gather function works the same as:
            # for i in range(len(all_q_values)):
            #   q_values.append(all_q_values[i][action_batch[i]])
            # but we cannot use such loop, because Variables are immutable,
            # so they don't have 'append' function etc.
            # squeezing magic needed for size mismatches, debug
            # yourself if you wonder why the're necessary
            q_values = all_q_values.\
                gather(1, action_batch.unsqueeze(1)).squeeze()

            # q_next_max = max{a}{Q(s_next,a)}
            # Note: We create new Variable after the first line. Why?
            # We used the network parameters to calculate
            # q_next_max, but we don't want the backward() function to
            # propagate twice into these parameters. Creating new Variable
            # 'cuts' this part of computational graph - prevents it.
            q_next_max = self.q_network(next_state_batch)
            q_next_max = Variable(q_next_max.data)
            q_next_max, _ = q_next_max.max(dim=1)

            # If the next state was terminal, we don't calculate the q value -
            # the target should be just = r
            q_t1_max_with_terminal = q_next_max.mul(1 - terminal_mask_batch)

            # Calculate the target = r + max{a}{Q(s_next,a)}
            targets = reward_batch + self.gamma * q_t1_max_with_terminal

            # calculate the loss (nll -> negative log likelihood/cross entropy)
            # TODO mse -> nll_loss.
            # and optimize the parameters
            self.optimizer.zero_grad()
            loss = nn.modules.SmoothL1Loss()(q_values, targets)
            loss.backward()
            self.optimizer.step()

            # NOTE: might be useful for debugging
            # print("Loss = ", loss.data.numpy()[0])

    def _get_next_action(self):
        """Returns next action given a state with use of the network

        Note: this should be epsilon-greedy

        """

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)

        # NOTE: is autograd disabled here? is it important?
        outputs = self.target_network.forward(
            Variable(torch.FloatTensor(self.current_state)))

        return np.argmax(outputs.data.numpy())

    def return_model_info(self):
        """
        Not sure what with this one for now.
        Depends on what pytorch uses to save the network models.
        Method should return the network params and all other params
        that we can reproduce the exact configuration later/save it to the db
        """
        # TODO implement me!
        pass

    def get_experience_batch(self):
        """
        Retrieves a random batch of transitions from memory and transforms it
        to separate PyTorch Variables.

        Transition is a tuple in form of:
        (state, action, reward, next_state, terminal_state)
        Returns:
            exp_batch - list of Variables in given order:
                [0] - input_state_batch
                [1] - action_batch
                [2] - reward_batch
                [3] - next_state_batch
                [4] - terminal_mask_batch
        """

        exp_batch = [0, 0, 0, 0, 0]
        transition_batch = random.sample(self.memory, self.batch_size)

        # Float Tensors
        for i in [0, 2, 3, 4]:
            exp_batch[i] = Variable(torch.Tensor(
                [x[i] for x in transition_batch]))

        # Long Tensor for actions
        exp_batch[1] = Variable(
            torch.LongTensor([int(x[1]) for x in transition_batch]))

        return exp_batch
