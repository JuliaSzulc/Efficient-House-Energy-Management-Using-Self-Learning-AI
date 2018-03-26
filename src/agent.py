import numpy as np
import random
import torch
from torch import autograd, nn, optim
import torch.nn.functional as F


class Net(torch.nn.Module):
    """
        Simple Neural Network.

    """
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
        return F.softmax(x, dim=1)


class Agent:
    """Reinforcement Learning agent.

    Agent interacts with the environment, gathering
    information about the reward for his previous actions,
    and observation of state transitions.
    """

    def __init__(self, env):
        self.env = env
        self.actions = self.env.get_actions()
        self.network = None
        self.current_state = None
        self.memory = []  # TODO - czy lista?
        self.gamma = 0
        self.epsilon = 0
        self.epsilon_decay = 0
        self.epsilon_min = 0
        self.batch_size = 0
        self.l_rate = 0.005
        self.optimizer = None
        self.initial_state = None
        # initialize parameters and the network
        self.reset()

    def reset(self):
        """Initialize the networks and other parameters"""
        self.initial_state = self.env.reset()
        input_size = len(self.initial_state)
        hidden1_size = 15
        hidden2_size = 10
        output_size = len(self.actions)
        self.network = Net(input_size, hidden1_size, hidden2_size, output_size)
        self.gamma = 0.95
        self.epsilon = 0.05  #
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 16
        self.l_rate = 0.005
        self.optimizer = optim.SGD(self.network.parameters(),
                                   lr=self.l_rate, momentum=0.9)
        pass

    def run(self):
        """Main agent's function. Performs the deep q-learning algorithm"""
        # TODO: reset musi zwrócić początkowy stan środowiska
        self.current_state = self.env.reset()
        total_reward = 0
        terminal_state = False
        while not terminal_state:
            action = self._get_next_action(self.current_state)
            next_state, reward, terminal_state = self.env.step(action)
            self.memory.append((self.current_state, action, reward,
                                next_state, terminal_state))
            self.current_state = next_state
            total_reward += reward
            self._train()

        return total_reward

    def _train(self):
        """
        Trains the underlying network with use of experience memory
        Note: this method does a training *step*, not whole training
        """
        # TODO implement me!
        if len(self.memory) > self.batch_size:
            self.optimizer.zero_grad()

            exp_batch = random.sample(self.memory, self.batch_size)
            input_state_batch = autograd.Variable(
                [x[0] for x in exp_batch], requires_grad=True)
            action_batch = autograd.Variable([x[1] for x in exp_batch])
            reward_batch = autograd.Variable([x[2] for x in exp_batch])
            next_state_batch = autograd.Variable([x[3] for x in exp_batch],
                                                 requires_grad=True)
            is_terminal_state_batch = autograd.Variable(
                [x[4] for x in exp_batch])

            # for each state in batch calc Q
            q_t0_values = self.network(input_state_batch)

            # for each next_state in batch calc max{a}{Q(s,a)}
            q_t1_max = torch.max(self.network(next_state_batch), 0)

            # if next_state - max{a}{Q(s,a)} should be zero
            q_t1_max_with_terminal = torch.mul(q_t1_max,
                                               is_terminal_state_batch)

            # now calc the targets for each batch (only for the action
            # taken in the batch!)
            targets = reward_batch + self.gamma * q_t1_max_with_terminal
            loss = torch.zeros(self.batch_size, len(self.actions))

            for sample, i in enumerate(loss):
                sample[action_batch[i]] = F.mse_loss(
                    q_t0_values[action_batch[i]], targets[i])

            # 6. train net with the error
            loss.backward()
            self.optimizer.step()

    def _get_next_action(self, state):
        """
        Returns next action given a state with use of the network
        Note: this should be epsilon-greedy
        """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        return np.argmax(self.network.forward(state)[0])

    def return_model_info(self):
        """
        Not sure what with this one for now.
        Depends on what pytorch uses to save the network models.
        Method should return the network params and all other params
        that we can reproduce the exact configuration later/save it to the db
        """
        # TODO implement me!
        pass
