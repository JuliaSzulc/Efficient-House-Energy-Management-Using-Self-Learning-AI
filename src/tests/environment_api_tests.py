import unittest
from unittest.mock import MagicMock
from src.environment import HouseEnergyEnvironment
import types


class BasicEnvironmentTestCase(unittest.TestCase):
    """Testing basic mocked environment usage"""

    def setUp(self):
        self.env = HouseEnergyEnvironment()

        def action(self):
            self.mocked_param += 1

        self.env.house.mocked_param = 0
        self.env.house.action_mocked = types.MethodType(action, self.env.house)

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
        self.assertTrue(isinstance(observation, dict),
                        "First value should be a dict "
                        "with new state observation")
        self.assertTrue(isinstance(reward, float) or isinstance(reward, int),
                        "Second value should be a number")
        self.assertTrue(isinstance(done, bool),
                        "Third value should be a bool")

    if __name__ == '__main__':
        unittest.main()
