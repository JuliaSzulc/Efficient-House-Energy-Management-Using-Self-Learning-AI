"""This module provides the clue of the project - RL agent.

It works with environment by taking actions and gaining observations and
reward, and its objective is to maximize cost function in a continuous
environment.

"""
import random
import sys
import os
import json
from collections import deque
from shutil import copyfile
import torch
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


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

    def __str__(self):
        return 'RL_Agent Object'

    def reset(self):
        """Initialize the networks, stats dictionary and other parameters

        Note that all of the network/learning parameters are either
        set as values from the configuration files or derived from environment
        characteristics - state and action space shapes.
        """

        self.initial_state = self.env.reset()
        self.actions = self.env.get_actions()

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
        self.memory = Memory(maxlen=self.config['memory_size'])
        self.double_dqn = self.config['double_dqn']
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
        """Main function of the agent, performs the episode and learning

        Lets the agent make steps in the environment, saves the observed
        transitions to memory, updates the stats and calls the _train() method.
        During the run, the target_network of the agent is slightly
        ("q_to_target_ratio") updated towards the q_network weights and is
        updated fully after every "target_network_update_freq".
        You can set these parameters in the configuration file.

        Please note that this method decays the epsilon value after the episode
        finishes.

        This method should be called in the outer training loop, as it performs
        one episode of training.
        """
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
            self.memory.append((self.current_state, action_index, reward,
                                next_state, terminal_state))

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
        # 'combined experience replay'
        # we use experience replay, but we put 2 last transitions in the batch
        # to overcome the problem of very slow training with big memory sizes
        # see paper "Deeper Look into Experience Replay" by G. Hinton
        transition_batch = random.sample(self.memory, self.batch_size - 2)
        transition_batch.append(self.memory[-2])
        transition_batch.append(self.memory[-1])

        # Float Tensors
        for i in [0, 2, 3, 4]:
            exp_batch[i] = Variable(torch.Tensor(
                [x[i] for x in transition_batch]))

        # Long Tensor for actions
        exp_batch[1] = Variable(
            torch.LongTensor([int(x[1]) for x in transition_batch]))

        return exp_batch

    # --- Define utility methods below ---

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


class AgentUtils:
    """Abstract class providing save and load methods for Agent objects"""

    @staticmethod
    def load(agent, model_id):
        """Loads network configuration and model

        Loads from file into the Agent's
        network fields.

        Args:
            agent(Agent): an Agent object, to whom we want to load
            model_id(int): id of model which we want to load

        """
        add_path = ''
        if 'tests' in os.getcwd():
            add_path = '../'
        conf_path = add_path + \
            'saved_models/model_{}/configuration.json'.format(model_id)
        model_path = add_path + \
            'saved_models/model_{}/network.pt'.format(model_id)

        # loading configuration file
        try:
            with open(conf_path) as config_file:
                agent.config = json.load(config_file)['agent']
            agent.reset()
        except FileNotFoundError as exc:
            print("Loading model failed. No model with given index, or no" + 
                  " configuration file. Error: \n")
            print(exc)
            sys.exit()

        # load network model
        try:
            agent.q_network.load_state_dict(torch.load(model_path))
            agent.target_network.load_state_dict(torch.load(model_path))
        except (RuntimeError, AssertionError) as exc:
            print('Error while loading model. Wrong network size, or not' +
                  ' an Agent? Aborting. Error:')
            print(exc)
            sys.exit()

    @staticmethod
    def save(model, rewards=None, old_id=None):
        """Save model, configuration file and training rewards

        Saving to files in the saved_models/{old_id} directory.

        Args:
            old_id(number): id of the model if it  was loaded, None otherwise
            model(torch.nn.Net): neural network torch model (q_network)
            rewards(list): list of total rewards for each episode, default None

        """
        add_path = ''
        if 'tests' in os.getcwd():
            add_path = '../'
        path = add_path + 'saved_models/model_'

        # create new directory with incremented id
        new_id = 0
        while True:
            if not os.path.exists(path + '{}'.format(new_id)):
                os.makedirs(path + '{}'.format(new_id))
                break
            new_id += 1

        # copy old rewards log to append new if model was loaded
        if old_id:
            try:
                copyfile(
                    path + '{}/rewards.log'.format(old_id),
                    path + '{}/rewards.log'.format(new_id))
            except FileNotFoundError:
                print('Warning: no rewards to copy found,\
                      but OLD ID is not None.')

        #  --- save new data
        # model
        torch.save(model.q_network.state_dict(),
                   path + '{}/network.pt'.format(new_id))

        # config
        config_path = add_path + '../configuration.json'
        if old_id:
            config_path = path + "{}/configuration.json".format(old_id)

        copyfile(config_path, path + "{}/configuration.json".format(new_id))

        if not rewards:
            return
        # rewards log
        with open(path + "{}/rewards.log".format(new_id), "a") as logfile:
            for reward in rewards:
                logfile.write("{}\n".format(reward))
        # rewards chart
        rewards = []
        for line in open(path + '{}/rewards.log'.format(new_id), 'r'):
            values = [float(s) for s in line.split()]
            rewards.append(values)
        avg_rewards = []
        for i in range(len(rewards) // (10 or 1)):
            avg_rewards.append(np.mean(rewards[10 * i: 10 * (i + 1)]))
        plt.plot(avg_rewards)
        plt.savefig(path + '{}/learning_plot.png'.format(new_id))
