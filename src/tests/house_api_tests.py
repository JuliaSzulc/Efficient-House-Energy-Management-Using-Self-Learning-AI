import unittest
from unittest.mock import MagicMock
import os, sys
from collections import OrderedDict
import math
import random

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from house import House


class HouseActionsDifferentTimeframes(unittest.TestCase):
    """Testing house actions on different timeframes"""

    def setUp(self):
        self.house_long = House(100)
        self.house_short = House(1)
        self.house_shortest = House(1 / 60)  # one second timeframe

        self.house_long.devices_settings['heating_lvl'] = 0
        self.house_short.devices_settings['heating_lvl'] = 0
        self.house_shortest.devices_settings['heating_lvl'] = 0

        self.house_long.devices_settings['cooling_lvl'] = 1
        self.house_short.devices_settings['cooling_lvl'] = 1
        self.house_shortest.devices_settings['cooling_lvl'] = 1

    def test_long_timeframe(self):
        """Test example action on long timeframe"""

        self.house_long.action_more_heating()
        self.assertEqual(self.house_long.devices_settings['heating_lvl'], 1)

        self.house_long.action_less_cooling()
        self.assertEqual(self.house_long.devices_settings['cooling_lvl'], 0)

    def test_short_timeframe(self):
        """Test example action on short timeframe"""

        self.house_short.action_more_heating()
        self.assertEqual(self.house_short.devices_settings['heating_lvl'], 0.2)

        self.house_short.action_less_cooling()
        self.assertEqual(self.house_short.devices_settings['cooling_lvl'], 0.8)

    def test_shortest_timeframe(self):
        """Test example action on shortest timeframe"""

        for _ in range(10 * 60 + 1):
            self.house_shortest.action_more_heating()
        self.assertEqual(self.house_shortest.devices_settings['heating_lvl'],
                         1)

        for _ in range(10 * 60 + 1):
            self.house_shortest.action_less_cooling()
        self.assertEqual(self.house_shortest.devices_settings['cooling_lvl'],
                         0)


class HouseActionsTestCase(unittest.TestCase):
    """Testing house actions"""

    def setUp(self):
        self.house = House(1)  # one minute timeframe
        self.house.influence = 0.2
        self.house.devices_settings = {
            'energy_src': 'grid',
            'cooling_lvl': 0.5,
            'heating_lvl': 0.5,
            'light_lvl': 0.5,
            'curtains_lvl': 0.5
        }
        self.house.battery['current'] = 0.5

        self.house_empty_battery = House(1)
        self.house_empty_battery.battery['current'] = 0.2
        self.house_empty_battery.devices_settings = {
            'energy_src': 'grid',
        }

    def test_action_sources(self):
        """Test changing between power sources"""

        self.house.action_source_battery()
        self.assertTrue(self.house.devices_settings['energy_src'], 'battery')

        self.house.action_source_grid()
        self.assertTrue(self.house.devices_settings['energy_src'], 'grid')

    def test_action_more_cooling(self):
        """Test more cooling"""

        self.house.action_more_cooling()
        self.assertEqual(self.house.devices_settings['cooling_lvl'], 0.7)

        for _ in range(3):
            self.house.action_more_cooling()
        self.assertEqual(self.house.devices_settings['cooling_lvl'], 1)

    def test_action_less_cooling(self):
        """Test less cooling"""

        self.house.action_less_cooling()
        self.assertEqual(self.house.devices_settings['cooling_lvl'], 0.3)

        for _ in range(3):
            self.house.action_less_cooling()
        self.assertEqual(self.house.devices_settings['cooling_lvl'], 0)

    def test_action_more_heating(self):
        """Test more heating"""

        self.house.action_more_heating()
        self.assertEqual(self.house.devices_settings['heating_lvl'], 0.7)

        for _ in range(3):
            self.house.action_more_heating()
        self.assertEqual(self.house.devices_settings['heating_lvl'], 1)

    def test_action_less_heating(self):
        """Test less heating"""

        self.house.action_less_heating()
        self.assertEqual(self.house.devices_settings['heating_lvl'], 0.3)

        for _ in range(3):
            self.house.action_less_heating()
        self.assertEqual(self.house.devices_settings['heating_lvl'], 0)

    def test_action_more_light(self):
        """Test more light"""

        self.house.action_more_light()
        self.assertEqual(self.house.devices_settings['light_lvl'], 0.6)

        for _ in range(6):
            self.house.action_more_light()
        self.assertEqual(self.house.devices_settings['light_lvl'], 1)

    def test_action_less_light(self):
        """Test less light"""

        self.house.action_less_light()
        self.assertEqual(self.house.devices_settings['light_lvl'], 0.4)

        for _ in range(6):
            self.house.action_less_light()
        self.assertEqual(self.house.devices_settings['light_lvl'], 0)

    def test_action_curtains_up(self):
        """Test curtains up"""

        self.house.action_curtains_up()
        self.assertEqual(self.house.devices_settings['curtains_lvl'], 0.4)

        for _ in range(6):
            self.house.action_curtains_up()
        self.assertEqual(self.house.devices_settings['curtains_lvl'], 0)

    def test_action_curtains_down(self):
        """Test curtains down"""

        self.house.action_curtains_down()
        self.assertEqual(self.house.devices_settings['curtains_lvl'], 0.6)

        for _ in range(6):
            self.house.action_curtains_down()
        self.assertEqual(self.house.devices_settings['curtains_lvl'], 1)


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
        self.house.daytime = 12 * 60  # default

        # define perfect conditions (so reward should be zero)
        self.house._calculate_energy_cost = MagicMock(return_value=0)
        self.house.user_requests = {
                'temp_desired': 21,
                'temp_epsilon': 0.5,
                'light_desired': 0.7,
                'light_epsilon': 0.05
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
        It shouldn't return positive values.
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

    def test_reward_decrease_with_energy_cost(self):
        """
        Energy cost is the base parameter and with cost increase,
        penalty should be bigger
        """
        reward = self.house.reward()
        self.house._calculate_energy_cost = MagicMock(return_value=100)
        self.assertLess(self.house.reward(), reward,
                        "Reward should decrease, cost parameter got worse!")
        self.house._calculate_energy_cost = MagicMock(return_value=0)
        self.house.inside_sensors = {
            'first': {
                'temperature': 20,
                'light': 0.7
            }
        }
        self.assertLess(self.house.reward(), reward,
                        "Reward should decrease, temperature got worse!")
        self.house.inside_sensors = {
            'first': {
                'temperature': 21,
                'light': 0.4
            }
        }
        self.assertLess(self.house.reward(), reward,
                        "Reward should decrease, light got worse!")


class HouseUpdateTestCase(unittest.TestCase):
    """Testing all the methods from update and update itself"""

    def setUp(self):
        self.house = House(1)

        self.sensor_out_info = {
            'daytime': 12 * 60,
            'actual_temp': 15,
            'light': 0.4,
            'illumination': 10000,
            'clouds': 0.4,
            'rain': 0,
            'wind': 0.7
        }

        self.house.update(self.sensor_out_info)

    def test_compare_daytime(self):
        self.assertEqual(self.house.daytime, self.sensor_out_info['daytime'],
                         "Daytime is different in house and outside sensor!")


class HouseEnergyCostTestCase(unittest.TestCase):
    """Testing energy cost calculation method"""

    def setUp(self):
        self.house = House(1)

        self.house.devices_settings = OrderedDict({
            'energy_src': 'battery',
            'cooling_lvl': 1.0,
            'heating_lvl': 1.0,
            'light_lvl': 1.0,
            'curtains_lvl': 1.0
        })

        self.house.battery['current'] = 0.001

    def test_change_source_if_not_enough_energy_in_battery(self):
        self.house._calculate_energy_cost()
        self.assertTrue(self.house.devices_settings['energy_src'] == 'grid')

    def test_energy_cost_not_negative(self):
        self.house.devices_settings = OrderedDict({
            'energy_src': 'grid',
            'cooling_lvl': 0,
            'heating_lvl': 0,
            'light_lvl': 0,
            'curtains_lvl': 0
        })

        self.assertTrue(self.house._calculate_energy_cost() >= 0,
                        "Energy cost should not be a negative number!\n"
                        "Settings: \n{}".format(self.house.devices_settings))

        for i in range(100):
            self.house.devices_settings = OrderedDict({
                'energy_src': 'grid',
                'cooling_lvl': random.random(),
                'heating_lvl': random.random(),
                'light_lvl': random.random(),
                'curtains_lvl': random.random()
            })
            self.assertTrue(self.house._calculate_energy_cost() >= 0,
                            "Energy cost should not be a negative number!\n"
                            "Settings:\n{}".format(self.house.devices_settings))


class BasicHouseTestCase(unittest.TestCase):
    """Testing house usage and methods"""

    def setUp(self):
        self.house = House(5)
        self.house.day_start = 7 * 60
        self.house.day_end = 24 * 60 - 5 * 60

    def test_get_inside_params(self):
        """
        Tests if returned dictionary is OrderedDict
        and if any inside dicitonaries are OrderedDict
        """
        inside_params = self.house.get_inside_params()

        is_ordered_dict = type(inside_params) is OrderedDict

        self.assertTrue(is_ordered_dict, "Returned dictionary has to be of "
                                         "OrderedDict type")

        for param in [d for d in inside_params if isinstance(d, dict)]:
            is_ordered_dict = type(param) is OrderedDict
            self.assertTrue(is_ordered_dict, "Inside dictionary has to be of "
                            "OrderedDict type")


if __name__ == '__main__':
    unittest.main()
