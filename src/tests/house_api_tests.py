import unittest
from unittest.mock import MagicMock
import os, sys

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


class BasicHouseTestCase(unittest.TestCase):
    """Testing house usage and methods"""

    def setUp(self):
        self.house = House(timeframe=5)
        self.house.day_start = 7 * 60
        self.house.day_end = 24 * 60 - 5 * 60

    def test_reward(self):
        """
        Tests whether the reward function:
        1. Returns 0 if the parameters are perfect (no difference between expected and desired, no energy cost)
        2. Returns bigger penalty when the difference increases
        """
        # case 1
        # given
        self.house._calculate_energy_cost = MagicMock(return_value=0)
        user_req_mock = {
                'temp_desired': 21,
                'temp_epsilon': 0.5,
                'light_desired': 0.7,
                'light_epsilon': 0.05
            }
        self.house._get_current_user_requests = MagicMock(return_value=user_req_mock)

        self.house.inside_sensors = {
            'first': {
                'temperature': 21,
                'light': 0.7
            }
        }

        # when
        reward = self.house.reward()

        # then
        self.assertEqual(reward, 0, "Reward should be zero, params are perfect!")

        # case 2
        # given
        self.house._calculate_energy_cost = MagicMock(return_value=100)
        self.house.inside_sensors = {
            'first': {
                'temperature': 19,
                'light': 0.5
            }
        }

        # when
        reward_2 = self.house.reward()

        # then
        # minus because rewards are negative
        self.assertGreater(-reward_2, -reward, "Reward should be bigger, parameters are worse.")

    def test_get_current_user_requests(self):
        """
        Tests if the method returns correct user requests based on the time of the day (night/day requests)
        """
        # given

        self.house.daytime = 2 * 60  # night

        # when
        requests = self.house._get_current_user_requests()

        # then
        self.assertDictEqual(requests, self.house.user_requests['night'], "Method returns user requests for the day, "
                                                                          "not for the night")
        # given
        self.house.daytime = 12 * 60  # day

        # when
        requests = self.house._get_current_user_requests()

        # then
        self.assertDictEqual(requests, self.house.user_requests['day'], "Method returns user requests for the night, "
                                                                        "not for the day")

    def test_calculate_penalty(self):
        """
        Tests if the method correctly calculates the penalty, given all the parameters
        """

        # given
        temp_current = 20.0
        temp_desired = 21.0
        temp_epsilon = 1.0
        power = 2

        # when
        penalty = House._calculate_penalty(temp_current, temp_desired, temp_epsilon, power)

        # then
        self.assertEqual(penalty, 0, "Penalty should be zero, because difference is not greater than epsilon!")

        # given
        temp_current = 100
        temp_desired = 50
        temp_epsilon = 1.0
        power = 2

        # when
        penalty = House._calculate_penalty(temp_current, temp_desired, temp_epsilon, power)

        # then
        self.assertEqual(penalty, 2500, "Penalty calculated incorrectly!")


if __name__ == '__main__':
    unittest.main()
