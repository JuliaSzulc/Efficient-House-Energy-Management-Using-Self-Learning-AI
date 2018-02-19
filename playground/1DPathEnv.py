import random

##################

# 1D Path Walk Enviroment 1D

# Parameters:
# - size = lenght of the state space
# - terminals = list of indexes of states that are terminal

# Rewards:
# - 1 if any terminal state is reached
# - 0 otherwise

##################


class 1DPathEnv:
    def __init__(self, is_cyclic, size, terminals):
        self.is_cyclic = is_cyclic
        self.size = size
        self.terminals = terminals
        self.actions = {'l': -1, 'r': 1}

    def sample_episode(self, policy):

        """Returns single episode generated with given policy. episode = list of tuples (s_i, a_i, r_(i-1))"""

        episode = list()
        state = random.choice(list(set(range(0, self.size)).difference(set(self.terminals))))
        reward, terminal = 0, False
        while not terminal:
            action = policy(state)
            episode.append((state, action, reward))
            reward, new_state, terminal = self.make_action(state, action)
            state = new_state
        return episode

    def make_action(self, state, action):

        """Returns reward from the acton, new state and is the new state a terminal state """

        new_state = self._get_new_state(state, action)
        if new_state in self.terminals:
            return 1, new_state, True
        else:
            return 0, new_state, False

    def _get_new_state(self, state, action):

        """Returns new state. If the enviroment is cyclic,
        state-action (0, l) returns state of index (size-1) etc."""

        new_state = state + self.actions[action]
        if self.is_cyclic:
            if new_state < 0:
                return self.size - 1
            if new_state >= self.size:
                return 0
        elif new_state < 0 or new_state >= self.size:
                return state

        return new_state
