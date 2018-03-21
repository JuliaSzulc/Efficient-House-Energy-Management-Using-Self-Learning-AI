import os
import sys
import unittest

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from agent import Agent
from environment import HouseEnergyEnvironment


class BasicAgentTestCase(unittest.TestCase):
    """Agent tests. Please follow test rules we defined for other classes"""

    def setUp(self):
        self.env = HouseEnergyEnvironment()
        self.agent = Agent(self.env)
