import unittest
from unittest.mock import MagicMock
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from environment import HouseEnergyEnvironment
import numpy as np
import types


class BasicEnvironmentTestCase(unittest.TestCase):
    """Testing basic mocked environment usage"""

    def setUp(self):
        self.env = HouseEnergyEnvironment()

        def action(self):
            self.mocked_param += 1

        self.env.house.mocked_param = 0
        self.env.house.action_mocked = types.MethodType(action, self.env.house)

    def test_serialize_state_values_on_full_episode(self):
        """test if returned state in step method has normalized values"""
        done = False
        while not done:
            observation, reward, done = self.env.step("action_mocked")
            observation = observation.tolist()
            self.assertTrue(all([0 <= x <= 1 for x in observation]),
                            "state is not serialized! observation vector:\n" +
                            " ".join([str(x) for x in observation]))

    def test_serialized_vector_length(self):
        """Test if vector have proper length"""
        proper = 21
        for _ in range(10):
            observation, reward, done = self.env.step("action_mocked")
            observation = observation.tolist()
            self.assertEqual(
                len(observation),
                proper,
                "If this is intentional, change this test case."
            )

    def test_get_action(self):
        """test getting mocked action"""

        actions = self.env.get_actions()
        self.assertTrue(
            "action_mocked" in actions,
            "failed at getting actions!"
        )

    def test_make_action_in_step(self):
        """test making mocked action"""
        self.env.step("action_mocked")

        self.assertEqual(
            self.env.house.mocked_param,
            1,
            "mocked action on environment failed!"
        )


class EnvironmentStepTestCase(unittest.TestCase):
    """ Testing environment's step function"""

    def setUp(self):
        self.env = HouseEnergyEnvironment()
        self.env.house.action_more_light = MagicMock()
        self.env.house.action_less_cooling = MagicMock()

    def test_step_chosen_action_called(self):
        """test calling chosen action once"""

        self.env.step("action_more_light")
        self.env.house.action_more_light.assert_called_once_with()

        self.env.step("action_less_cooling")
        self.env.house.action_less_cooling.assert_called_once_with()

    def test_step_amount_of_returned_values(self):
        """Test the amount of returned values"""

        values = self.env.step("action_more_light")
        self.assertEqual(len(values), 3, "Step should return 3 values")

    def test_step_type_of_returned_values(self):
        """Test the types of returned values"""

        observation, reward, done = self.env.step("action_more_light")
        self.assertTrue(isinstance(observation, np.ndarray),
                        "First value should be a 1-dim ndarray "
                        "with new state observation")
        self.assertTrue(isinstance(reward, float) or isinstance(reward, int),
                        "Second value should be a number")
        self.assertTrue(isinstance(done, bool),
                        "Third value should be a bool")


class EnvironmentStatisticsTestCase(unittest.TestCase):
    """ Testing environment's statistics functionality"""

    def setUp(self):
        self.stats_env = HouseEnergyEnvironment(collect_stats=True)
        self.no_stats_env = HouseEnergyEnvironment(collect_stats=False)

    def test_no_stats_env_returns_none(self):
        self.no_stats_env.reset()
        self.assertIsNone(self.no_stats_env.get_episode_stats())
        self.no_stats_env.step("action_nop")
        self.assertIsNone(self.no_stats_env.get_episode_stats())

    def test_stats_env_returns_none_after_reset(self):
        self.stats_env.reset()
        self.assertIsNone(self.stats_env.get_episode_stats())

    def test_stats_return_dict_after_step_taken(self):
        self.stats_env.reset()
        self.stats_env.step("action_nop")
        stats = self.stats_env.get_episode_stats()
        self.assertTrue(isinstance(stats, dict))

    def test_no_stats_env_not_calls_stats_update_on_step(self):
        self.no_stats_env.reset()
        self.no_stats_env._update_stats = MagicMock()
        self.no_stats_env.step("action_nop")
        self.no_stats_env._update_stats.assert_not_called()

    def test_stats_env_calls_stats_update_once_on_step(self):
        self.stats_env.reset()
        self.stats_env._update_stats = MagicMock()
        self.stats_env.step("action_nop")
        self.stats_env._update_stats.assert_called_once()


if __name__ == '__main__':
    unittest.main()
