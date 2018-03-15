import unittest
from unittest.mock import MagicMock
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from house import House


class BasicHouseTestCase(unittest.TestCase):
    """Testing house usage and methods"""

    def setUp(self):
        self.house = House()
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
