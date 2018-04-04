import unittest
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from agent import Agent
from environment import HouseEnergyEnvironment
import numpy as np


class TestEnv:
    """ Might be useful in future / side scripts for testing the agent manually
    with very simple env
    """

    def __init__(self):
        self.state = np.array([0.1, 0.2, 0.3, 0.4, 0.6])
        self.action_space = ['action1', 'action2', 'action3', 'action4']
        pass

    def reset(self):
        return self.state

    def get_actions(self):
        return self.action_space

    def step(self, action):
        return self.state, 5 if action == 'action1' else -5, False


class BasicAgentTestCase(unittest.TestCase):
    """ Agent tests. For now we don't unit test the methods like run or learn,
        as they will strongly depend on algorithms and mechanisms used.
    """

    def setUp(self):
        self.testEnv = TestEnv()
        self.agent = Agent(self.testEnv)

        self.agent.memory = [(np.array([0.1]), 0, -5,
                         np.array([0.1]), False),
                        (np.array([0.1]), 1, -5,
                         np.array([0.1]), False),
                        (np.array([0.1]), 2, -5,
                         np.array([0.1]), False),
                        (np.array([0.1]), 2, -5,
                         np.array([0.1]), False),
                        ]

    def test_get_experience_batch(self):
        self.agent.batch_size = 4
        exp_batch = self.agent.get_experience_batch()

        self.assertEqual(len(exp_batch), 5, "List of batches has wrong size")
        self.assertEqual(len(exp_batch[0].data), 4,
                         "Batch should have 4 entries")
