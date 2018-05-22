"""This module provides the clue of the project - RL agent.

It works with environment by taking actions and gaining observations and
reward, and its objective is to maximalize cost function in a continuous
environment.

"""
import random
import sys
import os
import json
import numpy as np
import torch
from collections import deque
from torch import optim, nn
from torch.autograd import Variable
import torch.nn.functional as F


class Net(torch.nn.Module):
    """Neural Network with variable layer sizes and 1 hidden layer."""

    def __init__(self, input_size, hidden1_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden1_size)
        self.fc2 = torch.nn.Linear(hidden1_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SumTree:
    """Structure used for storing priorities of the transitions.

    Used for PER (prioritized experience replay) implementation in Memory
    """
    pointer = 0

    def __init__(self, max_size):
        # number of leaves
        self.max_size = max_size
        # like a binary heap in an array
        # number-of-leaves + number-of-nodes = max_size + (max_size - 1)
        self.nodes = np.zeros(2 * max_size - 1)
        self.data = np.zeros(max_size, dtype=object)

    def update(self, index, priority):
        change = priority - self.nodes[index]
        self.nodes[index] = priority

        while index != 0:
            index = (index - 1) // 2
            self.nodes[index] += change

    def add(self, priority, data):
        index = self.pointer + self.max_size - 1
        self.data[self.pointer] = data
        self.update(index, priority)

    def get_leaves(self):
        return self.nodes[-self.max_size:]

    def get(self, value):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = 2 * parent_index + 2

            if left_child_index >= len(self.nodes):
                index = parent_index
                break
            else:
                if value <= self.nodes[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.nodes[left_child_index]
                    parent_index = right_child_index

        data_index = index - self.max_size + 1

        return (index, self.nodes[index], self.data[data_index])

    def get_priority_sum(self):
        return self.nodes[0]


class Memory:
    """Based on a SumTree, storing transitions batches for Agent.

    """

    # [0, 1] - prioritization factor that determines how much
    # prioritization is used( if 0 -> uniform case)
    alpha = 0.5
    # constant that prevents the transition to have 0 priority
    epsilon = 0.01

    def __init__(self, max_size):
        self.sum_tree = SumTree(max_size)
        self.len = 0

    def add(self, transition, error):
        priority = self.get_priority(error)
        self.sum_tree.add(priority, transition)

        self.len += 1

    def get_priority(self, error):
        return (abs(error) + self.epsilon) ** self.alpha

    def update(self, index, error):
        priority = self.get_priority(error)
        self.sum_tree.update(index, priority)

    def sample(self, batch_size):
        batch = []
        indexes = []
        priority_segment = self.sum_tree.get_priority_sum() / batch_size

        for i in range(batch_size):
            a = i * priority_segment
            b = (i + 1) * priority_segment
            value = random.uniform(a, b)

            (index, priority, data) = self.sum_tree.get(value)
            batch.append(data)
            indexes.append(index)

        return batch, indexes


class Agent:
    """Reinforcement Learning agent.

    Agent interacts with the environment, gathering
    information about the reward for his previous actions,
    and observation of state transitions.

    """

    def __init__(self, env, conf=None):
        self.config = conf
        if not conf:
            add_path = ''
            if 'tests' in os.getcwd():
                add_path = '../'
            with open(add_path + '../configuration.json') as config_file:
                self.config = json.load(config_file)['agent']

        self.env = env
        self.actions = None

        self.q_network = None
        self.target_network = None

        self.initial_state = None
        self.current_state = self.env.reset()

        self.memory = None
        self.double_dqn = None
        self.gamma = 0
        self.epsilon = 0
        self.epsilon_decay = 0
        self.epsilon_min = 0
        self.batch_size = 0
        self.l_rate = 0
        self.optimizer = None
        self.train_freq = 0

        self.stats = dict()

        self.reset()

    def reset(self):
        """Initialize the networks and other parameters"""
        self.initial_state = self.env.reset()
        self.actions = self.env.get_actions()

        # initialize stats dictionary
        for a in self.actions:
            self.stats[a] = {'count': 0,
                             'total_reward': 0}

        input_size = len(self.initial_state)
        hidden1_size = self.config['hidden_layer_size']
        output_size = len(self.actions)

        self.q_network = Net(
            input_size,
            hidden1_size,
            output_size
        )
        self.target_network = Net(
            input_size,
            hidden1_size,
            output_size
        )
        self.memory = Memory(self.config['memory_size'])
        self.double_dqn = self.config['double_dqn']git
        self.gamma = self.config['gamma']
        self.epsilon = self.config['epsilon']
        self.epsilon_decay = self.config['epsilon_decay']
        self.epsilon_min = self.config['epsilon_min']
        self.batch_size = self.config['batch_size']
        self.l_rate = self.config['learning_rate']
        self.optimizer = optim.SGD(self.q_network.parameters(),
                                   lr=self.l_rate,
                                   momentum=self.config['sgd_momentum'])

        self.train_freq = self.config['training_freq']

    def run(self):
        """Main agents function. Performs the deep q-learning algorithm"""
        counter = 0
        self.current_state = self.env.reset()
        total_reward = 0
        terminal_state = False
        for a in self.actions:
            self.stats[a] = {'count': 0,
                             'total_reward': 0}

        while not terminal_state:
            action_index = \
                self._get_next_action_epsilon_greedy(self.current_state)

            next_state, reward, terminal_state = \
                self.env.step(self.actions[action_index])

            if reward < self.config['reward_clip']:
                reward = self.config['reward_clip']

            self._update_stats(action_index)
            self.append_sample_to_memory(self.current_state, action_index,
                                         reward, next_state, terminal_state)

            self.current_state = next_state
            total_reward += reward

            counter = (counter + 1) % self.config['target_network_update_freq']
            train_episode = not counter % self.train_freq
            if train_episode:
                self._train()

            # Update the target network:
            qt = self.config["q_to_target_ratio"]
            if counter == 0:
                qt = 1.0
            for target_param, q_param in zip(
                    self.target_network.parameters(),
                    self.q_network.parameters()):
                target_param.data.copy_(q_param.data * qt
                                        + target_param.data * (1.0 - qt))

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return total_reward

    def _train(self):
        """Trains the underlying q_network with use of experience memory.

        Training can use the advantages of DoubleDQN.
        Note: this method does a training *step*, not the whole training.

        """

        if self.memory.len > self.batch_size:
            exp_batch, indexes = self.get_experience_batch()

            input_state_batch = exp_batch[0]
            action_batch = exp_batch[1]
            reward_batch = exp_batch[2]
            next_state_batch = exp_batch[3]
            terminal_mask_batch = exp_batch[4]

            all_q_values = self.q_network(input_state_batch)

            q_values = all_q_values. \
                gather(1, action_batch.unsqueeze(1)).squeeze()

            if self.double_dqn:
                # --- Formula:
                # Ydouble = r +
                # Q(s_next, argmax{a}{Q(s_next,a; q_network)}; target_network)

                # --- step by step explanation:
                # q2 = Q(s_next, a; q_network)
                # action = argmax{a}{q2}
                # q1 = Q(s_next, action; target_network)
                # Ydouble = r + gamma * q1

                q2 = self.q_network(next_state_batch)
                q2 = Variable(q2.data)
                _, actions = q2.max(dim=1)

                q1 = self.target_network(next_state_batch)
                q1 = Variable(q1.data)
                q1 = q1.gather(1, actions.unsqueeze(1)).squeeze()

                q_t1_max_with_terminal = q1.mul(1 - terminal_mask_batch)
            else:
                q_next_max = self.q_network(next_state_batch)
                q_next_max = Variable(q_next_max.data)
                q_next_max, _ = q_next_max.max(dim=1)
                q_t1_max_with_terminal = q_next_max.mul(1 - terminal_mask_batch)

            targets = reward_batch + self.gamma * q_t1_max_with_terminal

            errors = torch.abs(q_values - targets).data.numpy()

            for i in range(self.batch_size):
                index = indexes[i]
                self.memory.update(index, errors[i])

            self.optimizer.zero_grad()
            loss = nn.modules.SmoothL1Loss()(q_values, targets)
            loss.backward()
            for param in self.q_network.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def get_next_action_greedy(self, state):
        """
        Returns next action given a state with use of the target network
        using a greedy policy. This function should be used if an outside
        object wants to know the agent's action for the state.

        Args:
            state(dict): state information used as an input for the network

        """

        outputs = self.target_network.forward(
            Variable(torch.FloatTensor(state)))
        return np.argmax(outputs.data.numpy())

    def _get_next_action_epsilon_greedy(self, state):
        """
        Returns next action given a state with use of the target network
        using an epsilon greedy policy. Epsilon is the probability of choosing
        a random action.

        Args:
            state(dict): state information used as an input for the network

        """

        if np.random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        else:
            return self.get_next_action_greedy(state)

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
        transition_batch, indexes = self.memory.sample(self.batch_size)

        # Float Tensors
        for i in [0, 2, 3, 4]:
            exp_batch[i] = Variable(torch.Tensor(
                [x[i] for x in transition_batch]))

        # Long Tensor for actions
        exp_batch[1] = Variable(torch.LongTensor(
            [int(x[1]) for x in transition_batch]))

        return exp_batch, indexes

    # --- Define utility methods below ---

    def load_network_model(self, path):
        """
        Loads network model from given file into the Agent's network fields.

        Args:
            path(str): path to file

        """
        try:
            self.q_network.load_state_dict(torch.load(path))
            self.target_network.load_state_dict(torch.load(path))
        except RuntimeError:
            print('Error while loading model. Wrong network size? Aborting')
            sys.exit()

    def load_config(self, path):
        """
        Loads configuration from given file.

        Args:
            path(str): path to file

        """
        try:
            with open('../configuration.json') as config_file:
                self.config = json.load(config_file)['agent']
            self.reset()
        except FileNotFoundError:
            print('Configuration file doesnt exist!')
            sys.exit()

    def _update_stats(self, action_index):
        """Updates agent statistics"""
        action = self.actions[action_index]
        self.stats[action]['count'] += 1

    def get_episode_stats(self):
        """Returns statistics about agent's actions.

        Currently the method returns a dict with most commonly chosen action
        name and number of times it was taken since the last reset() call.

        Returns:
            dict of statistics about agent's actions
        """
        most_common = max(self.stats.items(), key=lambda item:
                          item[1]['count'])

        aggregated = {
            'most common action': (
                most_common[0],
                most_common[1]['count']
            )
        }

        return aggregated

    def append_sample_to_memory(self, current_state, action_index, reward,
                                next_state, terminal_state):
        target = self.q_network(Variable(
            torch.FloatTensor(current_state))).data
        q = target[action_index]
        target_val = self.target_network(Variable(
            torch.FloatTensor(next_state))).data
        if terminal_state:
            target[action_index] = reward
        else:
            target[action_index] = reward + self.gamma * torch.max(
                target_val)

        error = abs(q - target[action_index])

        self.memory.add((current_state, action_index, reward, next_state,
                         terminal_state), error)
