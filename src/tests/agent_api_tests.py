import unittest
from src.agent import Agent
from src.environment import HouseEnergyEnvironment


class BasicAgentTestCase(unittest.TestCase):
    """Agent tests. Please follow test rules we defined for other classes"""

    def setUp(self):
        self.env = HouseEnergyEnvironment()
        self.agent = Agent(self.env)
