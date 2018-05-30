import unittest
import shutil
import json
import torch
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from agent import Agent, AgentUtils
from environment import HouseEnergyEnvironment
import numpy as np

# TODO: write some decent Agent tests.

# class TestEnv:
    # """ Might be useful in future / side scripts for testing the agent manually
    # with very simple env
    # """

    # def __init__(self):
        # self.state = np.array([0.1, 0.2, 0.3, 0.4, 0.6])
        # self.action_space = ['action1', 'action2', 'action3', 'action4']
        # pass

    # def reset(self):
        # return self.state

    # def get_actions(self):
        # return self.action_space

    # def step(self, action):
        # return self.state, 5 if action == 'action1' else -5, False


# class BasicAgentTestCase(unittest.TestCase):
    # """ Agent tests. For now we don't unit test the methods like run or learn,
        # as they will strongly depend on algorithms and mechanisms used.
    # """

    # def setUp(self):
        # self.testEnv = TestEnv()
        # self.agent = Agent(self.testEnv)

        # self.agent.memory = [(np.array([0.1]), 0, -5,
                         # np.array([0.1]), False),
                        # (np.array([0.1]), 1, -5,
                         # np.array([0.1]), False),
                        # (np.array([0.1]), 2, -5,
                         # np.array([0.1]), False),
                        # (np.array([0.1]), 2, -5,
                         # np.array([0.1]), False),
                        # ]

    # def test_get_experience_batch(self):
        # self.agent.batch_size = 4
        # exp_batch = self.agent.get_experience_batch()

        # self.assertEqual(len(exp_batch), 5, "List of batches has wrong size")
        # self.assertEqual(len(exp_batch[0].data), 4,
                         # "Batch should have 4 entries")


class AgentUtilsTestCase(unittest.TestCase):
    def setUp(self):
        env = HouseEnergyEnvironment()
        self.agent = Agent(env=env)
        self.script_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # -- save
        self.save_id = 0

        # paths
        self.save_path = self.script_path + \
            '/../saved_models/model_{}'.format(self.save_id)
        self.temp_path = self.script_path + \
            '/../saved_models/model_TEMP_FROM_TEST'

        # clean up
        if os.path.exists(self.temp_path):
            shutil.rmtree(self.temp_path)
        if os.path.exists(self.save_path):  # save if real model exists
            os.rename(self.save_path, self.temp_path)

        # -- load
        self.load_id = 999

        self.load_path = self.script_path + \
            '/../saved_models/model_{}'.format(self.load_id)

        # clean up
        if os.path.exists(self.load_path):
            shutil.rmtree(self.load_path)
        os.mkdir(self.load_path)

        torch.save(self.agent.q_network.state_dict(), self.load_path +
                   '/network.pt')

        shutil.copyfile(
            self.script_path + '/../../configuration.json',
            self.load_path + "/configuration.json")

        filename = self.load_path + '/configuration.json'
        with open(filename, 'r') as f:
            data = json.load(f)
            data['test'] = 'TRUE'

        os.remove(filename)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def tearDown(self):
        # restore if temp model exists
        if os.path.exists(self.temp_path):
            if os.path.exists(self.save_path):
                shutil.rmtree(self.save_path)
            os.rename(self.temp_path, self.save_path)

        # remove temp load directory
        if os.path.exists(self.load_path):
            shutil.rmtree(self.load_path)

    def test_load_given_id_test(self):
        """Load agent with given id"""
        AgentUtils.load(self.agent, self.load_id)
        with open(self.load_path + '/configuration.json') as config_file:
            config = json.load(config_file)
            self.assertTrue(config['test'])

    def test_save_index_0(self):
        """Save model to first free index, knowing that 0 is free"""

        AgentUtils.save(self.agent, rewards=None, old_id=self.load_id)
        self.assertTrue(os.path.exists(self.save_path))
        with open(self.save_path + '/configuration.json') as config_file:
            config = json.load(config_file)
            self.assertTrue(config['test'])


if __name__ == '__main__':
    unittest.main()
