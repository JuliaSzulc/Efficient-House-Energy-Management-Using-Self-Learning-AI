import unittest
from unittest.mock import patch
import os, sys
import json
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from environment import HouseEnergyEnvironment
from agent import Agent
from main import main as my_main


class BasicSubjectListenerTestCase(unittest.TestCase):
    """Testing observer/listener model and basic classes structure"""

    def setUp(self):
        self.env = HouseEnergyEnvironment()
        self.agent = Agent(env=self.env)
        self.my_main = my_main

        add_path = ''
        if 'tests' in os.getcwd():
            add_path = '../'
        with open(add_path + '../configuration.json') as config_file:
            self.config = json.load(config_file)
        self.config['main']['training_episodes'] = 1
        self.config['main']['save_experiment'] = False
        self.config['main']['print_stats'] = False
        self.config['main']['make_total_reward_plot'] = False
        self.config['main']['load_agent_model'] = False

    def test_agent_module(self):
        """Test agent.py consistency - run learning for three episodes"""
        for i in range(3):
            self.agent.run()

    def test_main_module(self):
        """Test main.py consistency - run for one episode"""
        self.my_main(config=self.config)


if __name__ == "__main__":
    unittest.main()
