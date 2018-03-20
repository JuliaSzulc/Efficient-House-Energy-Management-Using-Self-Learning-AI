import unittest
from unittest.mock import MagicMock
import os, sys

import math

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from house import House


class HouseActionsDifferentTimeframes(unittest.TestCase):
    """Testing house actions on different timeframes"""

    def setUp(self):
        self.house_long = House(100)
        self.house_short = House(1)
        self.house_shortest = House(1/60) # one second timeframe

        self.house_long.current_settings['heating_lvl'] = 0
        self.house_short.current_settings['heating_lvl'] = 0
        self.house_shortest.current_settings['heating_lvl'] = 0

        self.house_long.current_settings['cooling_lvl'] = 1
        self.house_short.current_settings['cooling_lvl'] = 1
        self.house_shortest.current_settings['cooling_lvl'] = 1

    def test_long_timeframe(self):
        """Test example action on long timeframe"""

        self.house_long.action_more_heating()
        self.assertEqual(self.house_long.current_settings['heating_lvl'], 1)

        self.house_long.action_less_cooling()
        self.assertEqual(self.house_long.current_settings['cooling_lvl'], 0)

    def test_short_timeframe(self):
        """Test example action on short timeframe"""

        self.house_short.action_more_heating()
        self.assertEqual(self.house_short.current_settings['heating_lvl'], 0.2)

        self.house_short.action_less_cooling()
        self.assertEqual(self.house_short.current_settings['cooling_lvl'], 0.8)

    def test_shortest_timeframe(self):
        """Test example action on shortest timeframe"""

        for _ in range(5 * 60 + 1):
            self.house_shortest.action_more_heating()
        self.assertEqual(self.house_shortest.current_settings['heating_lvl'], 1)

        for _ in range(5 * 60 + 1):
            self.house_shortest.action_less_cooling()
        self.assertEqual(self.house_shortest.current_settings['cooling_lvl'], 0)

class HouseActionsTestCase(unittest.TestCase):
    """Testing house actions"""

    def setUp(self):
        self.house = House(1) # one minute timeframe
        self.house.current_settings = {
            'energy_src': 'grid',
            'cooling_lvl': 0.5,
            'heating_lvl': 0.5,
            'light_lvl': 0.5,
            'curtains_lvl': 0.5
        }

    def test_action_sources(self):
        """Test changing between power sources"""

        self.house.action_source_battery()
        self.assertTrue(self.house.current_settings['energy_src'], 'battery')

        self.house.action_source_grid()
        self.assertTrue(self.house.current_settings['energy_src'], 'grid')

    def test_action_more_cooling(self):
        """Test more cooling"""

        self.house.action_more_cooling()
        self.assertEqual(self.house.current_settings['cooling_lvl'], 0.7)

        for _ in range(3):
            self.house.action_more_cooling()
        self.assertEqual(self.house.current_settings['cooling_lvl'], 1)

    def test_action_more_heating(self):
        """Test more heating"""

        self.house.action_more_heating()
        self.assertEqual(self.house.current_settings['heating_lvl'], 0.7)

        for _ in range(3):
            self.house.action_more_heating()
        self.assertEqual(self.house.current_settings['heating_lvl'], 1)

    def test_action_more_light(self):
        """Test more light"""

        self.house.action_more_light()
        self.assertEqual(self.house.current_settings['light_lvl'], 0.7)

        for _ in range(3):
            self.house.action_more_light()
        self.assertEqual(self.house.current_settings['light_lvl'], 1)

    def test_action_less_cooling(self):
        """Test less cooling"""

        self.house.action_less_cooling()
        self.assertEqual(self.house.current_settings['cooling_lvl'], 0.3)

        for _ in range(3):
            self.house.action_less_cooling()
        self.assertEqual(self.house.current_settings['cooling_lvl'], 0)

    def test_action_less_heating(self):
        """Test less heating"""

        self.house.action_less_heating()
        self.assertEqual(self.house.current_settings['heating_lvl'], 0.3)

        for _ in range(3):
            self.house.action_less_heating()
        self.assertEqual(self.house.current_settings['heating_lvl'], 0)

    def test_action_less_light(self):
        """Test less light"""

        self.house.action_less_light()
        self.assertEqual(self.house.current_settings['light_lvl'], 0.3)

        for _ in range(3):
            self.house.action_less_light()
        self.assertEqual(self.house.current_settings['light_lvl'], 0)

    def test_action_curtains_up(self):
        """Test curtains up"""

        self.house.action_curtains_up()
        self.assertEqual(self.house.current_settings['curtains_lvl'], 0.3)

        for _ in range(3):
            self.house.action_curtains_up()
        self.assertEqual(self.house.current_settings['curtains_lvl'], 0)

    def test_action_curtains_down(self):
        """Test curtains down"""

        self.house.action_curtains_down()
        self.assertEqual(self.house.current_settings['curtains_lvl'], 0.7)

        for _ in range(3):
            self.house.action_curtains_down()
        self.assertEqual(self.house.current_settings['curtains_lvl'], 1)


class HouseRewardTestCase(unittest.TestCase):
    """
    Testing the reward function (house is the source of it)
    Note: 'reward' is used as the standard reinforcement learning name,
    but the function works as a penalty.
    """

    def setUp(self):
        self.house = House(timeframe=5)
        self.house.day_start = 7 * 60
        self.house.day_end = 24 * 60 - 5 * 60
        self.house.daytime = 12 * 60 # default

        # define perfect conditions (so reward should be zero)
        self.house._calculate_energy_cost = MagicMock(return_value=0)
        self.house.user_requests = {
            'day' : {
                'temp_desired': 21,
                'temp_epsilon': 0.5,
                'light_desired': 0.7,
                'light_epsilon': 0.05
            }
        }
        self.house.inside_sensors = {
            'first': {
                'temperature': 21,
                'light': 0.7
            }
        }

    def test_reward_for_perfect_conditions(self):
        """
        Reward should be zero as the factors are perfect
        """
        reward = self.house.reward()
        self.assertEqual(reward, 0,
                         "Reward should be zero, factors are perfect!")

    def test_reward_returns_nonpositive_values(self):
        """
        The reward in the simulator is modeled as a penalty.
        It shouldn't return positive values
        """
        testing_pairs = ((-40, 0), (0, 0), (10, 0), (15, 0.02), (15, 0.5),
                         (21, 0.4), (264, 0.99), (math.pi, 1))
        for temp, light in testing_pairs:
            self.house.inside_sensors = {
                'first': {
                    'temperature': temp,
                    'light': light
                }
            }
            reward = self.house.reward()
            self.assertLessEqual(reward, 0, "Reward shouldn't be positive!")

    def test_reward_increase_with_energy_cost(self):
        """
        Energy cost is the base parameter and with cost increase,
        penalty should be bigger
        """
        reward = self.house.reward()
        self.house._calculate_energy_cost = MagicMock(return_value=100)
        reward_2 = self.house.reward()
        self.assertLess(reward_2, reward,
                        "Reward should be bigger, parameters are worse.")


class BasicHouseTestCase(unittest.TestCase):
    """Testing house usage and methods"""

    def setUp(self):
        self.house = House(5)
        self.house.day_start = 7 * 60
        self.house.day_end = 24 * 60 - 5 * 60

    def test_get_current_user_requests(self):
        """
        Tests get current user requests
        """
        self.house.daytime = 2 * 60  # night
        requests = self.house._get_current_user_requests()
        self.assertDictEqual(requests, self.house.user_requests['night'],
                             "Method returns user requests for the day, "
                             "not for the night")

        self.house.daytime = 12 * 60  # day
        requests = self.house._get_current_user_requests()
        self.assertDictEqual(requests, self.house.user_requests['day'],
                             "Method returns user requests for the night, "
                             "not for the day")

    def test_calculate_penalty(self):
        """
        Tests if the method correctly calculates the penalty
        """
        penalty = House._calculate_penalty(20.0, 21.0, 1.0, 2)
        self.assertEqual(penalty, 0, "Penalty should be zero, "
                                     "difference is not greater than epsilon")

        penalty = House._calculate_penalty(100, 50, 1.0, 2)
        self.assertEqual(penalty, 2500, "Penalty calculated incorrectly!")


if __name__ == '__main__':
    unittest.main()
