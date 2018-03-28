import unittest
from unittest.mock import MagicMock
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from world import World

class SunPowerInDifferentStates(unittest.TestCase):
    """ Testing sun power changes while day time is changing """

    def setUp(self):
        self.world = World()

    def test_sun_power_in_our_shining_time(self):
        for _ in range(1000):
            self.world.step()

            if 300 < self.world.daytime < 1140:
                self.assertTrue(0 < self.world.weather['sun'] <= 1, "Sun should shining now.")

    def test_sun_power_outside_our_shining_time(self):
        for _ in range(1000):
            self.world.step()

            if not (300 <= self.world.daytime <= 1140):
                self.assertTrue(self.world.weather['sun'] == 0, "Sun should not shining now.")

    def test_wind_power_in_different_stapes(self):
        for _ in range(1000):
            self.world.step()

            self.assertTrue(0 <= self.world.weather['wind'] <= 1, "Wind power incorrect value.")

    def test_clouds_interact_with_wind(self):
        for _ in range(1000):
            self.world.step()

            if self.world.weather['wind']:
                self.assertTrue(0 <= self.world.weather['clouds'] < 0.6, "Clouds value calculated with wind incorrect.")

    def test_clouds_interact_without_wind(self):
        for _ in range(1000):
            self.world.step()

            if self.world.weather['wind'] == 0:
                self.assertTrue(0 <= self.world.weather['clouds'] <= 0.6,
                                "Clouds value calculated without wind incorrect.")

    def test_rain_not_falling(self):
        temp_list = [0, 0.2, 0.39]

        for value in temp_list:
            self.world.weather['clouds'] = value
            self.world._calculate_rain()
            self.assertEqual(self.world.weather['rain'], 0, "Incorrect rain value - rain should be not falling.")

    def test_rain_falling(self):
        temp_list = [0.4, 0.41, 0.6]

        for value in temp_list:
            self.world.weather['clouds'] = value
            self.world._calculate_rain()
            self.assertEqual(self.world.weather['rain'], 1, "Incorrect rain value - rain should be falling.")

    def test_light_value(self):
        self.world.weather['sun'] = 0.4
        self.world.weather['clouds'] = 0.5
        self.world._calculate_light()
        self.assertEqual(self.world.weather['light'], 0.2, "Incorrect light value.")

    def test_temperature_value(self):
        for _ in range(1000):
            self.world.step()

            self.assertTrue(6.5 <= self.world.weather['temp'] <= 30.5,
                            "Clouds value calculated without wind incorrect.")

    def test_weather_meaning_scale(self):
        self.assertEqual(self.world.previous_weather_meaning + self.world.current_weather_meaning, 1,
                         "Incorrect weather meaning scale values.")


if __name__ == '__main__':
    unittest.main()