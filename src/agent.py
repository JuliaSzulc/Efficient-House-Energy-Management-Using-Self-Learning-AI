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
import os


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
        self.current_state = self.env.reset()

        self.memory = Memory(maxlen=5000)
        self.gamma = 0
        self.epsilon = 0
        self.epsilon_decay = 0
        self.epsilon_min = 0
        self.batch_size = 0
        self.l_rate = 0
        self.optimizer = None

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
        self.epsilon = 0.1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.001
        self.batch_size = 16
        self.l_rate = 0.01
        self.optimizer = optim.Adagrad(self.q_network.parameters(),
                                       lr=self.l_rate)

    def run(self):
        """Main agent's function. Performs the deep q-learning algorithm"""
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

            if reward < -2:
                reward = -2

            self._update_stats(action_index, reward)
            self.memory.append((self.current_state, action_index, reward,
                                next_state, terminal_state))

            self.current_state = next_state
            total_reward += reward
            self._train()

            # Update the target network:
            qt = 0.2  # q to target ratio
            for target_param, q_param in zip(self.target_network.parameters(),
                                             self.q_network.parameters()):
                target_param.data.copy_(q_param.data * qt
                                        + target_param.data * (1.0 - qt))

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return total_reward

    def _train(self):
        """Trains the underlying q_network with use of experience memory

        Note: this method does a training *step*, not whole training

        """

        if len(self.memory) > self.batch_size:
            exp_batch = self.get_experience_batch()

            input_state_batch = exp_batch[0]
            action_batch = exp_batch[1]
            reward_batch = exp_batch[2]
            next_state_batch = exp_batch[3]
            terminal_mask_batch = exp_batch[4]

            all_q_values = self.q_network(input_state_batch)

            q_values = all_q_values. \
                gather(1, action_batch.unsqueeze(1)).squeeze()

            q_next_max = self.q_network(next_state_batch)
            q_next_max = Variable(q_next_max.data)
            q_next_max, _ = q_next_max.max(dim=1)

            q_t1_max_with_terminal = q_next_max.mul(1 - terminal_mask_batch)

            targets = reward_batch + self.gamma * q_t1_max_with_terminal

            self.optimizer.zero_grad()
            loss = nn.modules.SmoothL1Loss()(q_values, targets)
            loss.backward()
            self.optimizer.step()

    def get_next_action_greedy(self, state):
        """
        Returns next action given a state with use of the target network
        using a greedy policy. This function should be used if an outside
        object wants to know the agent's action for the state - not the
        _get_next_action_epsilon_greedy function!

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

    def save_model_info(self):
        """
        Method saves all networks models info to a specific files in
        saved_models directory.

        """
        # TODO: In future save to database

        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')

        new_index = 0
        while True:
            if not os.path.isfile(
                    'saved_models/agent_model_{}.pt'.format(new_index)):
                break
            new_index += 1

        torch.save(self.q_network.state_dict(),
                   'saved_models/agent_model_{}.pt'.format(new_index))

    def load_model_info(self, model_id):
        """
        Loads the given model to the Agent's network fields.

        Args:
            model_id(number): model's number used to find the corresponding file

        """
        # TODO: In future load from database

        try:
            if os.path.isfile(
                    'saved_models/agent_model_{}.pt'.format(model_id)):
                self.q_network. \
                    load_state_dict(torch.load('saved_models/agent_model_{}.pt'.
                                               format(model_id)))
                self.target_network = self.q_network
            else:
                print('[Error] No model with entered index.\n'
                      'Any models have been loaded.\n'
                      'Exiting...')
                raise SystemExit

        except RuntimeError:
            print('[Error] Oops! RuntimeError occurred while loading model.\n'
                  'Check if your saved model data is up to date.\n'
                  'Maybe it fetch different network size?\n'
                  'Exiting...')
            raise SystemExit

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

    def _update_stats(self, action_index, reward):
        action = self.actions[action_index]
        self.stats[action]['count'] += 1
        self.stats[action]['total_reward'] += reward

    def get_episode_stats(self):
        most_common = max(self.stats.items(), key=lambda item:
        item[1]['count'])

        least_common = min(self.stats.items(), key=lambda item:
        item[1]['count'])

        best_mean_reward = max(self.stats.items(), key=lambda item:
        item[1]['total_reward'] /
        (item[1]['count'] or 1))

        best_total_reward = max(self.stats.items(), key=lambda item:
        item[1]['total_reward'])

        aggregated = {
            'most common action': (
                most_common[0],
                most_common[1]['count']
            ),
            'least common action': (
                least_common[0],
                least_common[1]['count']
            ),
            'action with best avg reward': (
                best_mean_reward[0],
                best_mean_reward[1]['total_reward']
                / (best_mean_reward[1]['count'] or 1)
            ),
            'action with best total reward': (
                best_total_reward[0],
                best_total_reward[1]['total_reward']
            ),
            'avg total reward': sum(
                [i['total_reward'] for k, i in self.stats.items()]
            ) / (len(self.actions) or 1)
        }

        return aggregated
