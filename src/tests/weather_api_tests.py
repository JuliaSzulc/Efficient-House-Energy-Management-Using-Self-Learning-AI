import unittest
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from world import World
from datetime import timedelta
import random


class WeatherOnTimeTestCase(unittest.TestCase):
    """ Testing weather with continuous time on different time steps """

    def setUp(self):
        self.worlds = []

        timesteps = [0.5, 1, 5, 15, 30]
        #should work on random timesteps
        # timesteps += [random.randrange(1, 30) for _ in range(10)]
        for timestep in timesteps:
            self.worlds.append(World(timestep))

        self.temp_for_rain_not_falling = [0, 0.2, 0.39]
        self.temp_for_rain_falling = [0.4, 0.41, 0.6]

    def test_sun_power_in_our_shining_time(self):
        for world in self.worlds:
            for _ in range(1000):
                world.step()

                if 300 < world.daytime < 1140:
                    self.assertTrue(0 < world.weather['sun'] <= 1,
                                    "Sun should shine now. timestep = "
                                    + str(world.time_step_in_minutes)
                                    + " current value = "
                                    + str(world.weather['sun']))

    def test_sun_power_outside_our_shining_time(self):
        for world in self.worlds:
            for _ in range(1000):
                world.step()

                if not 300 <= world.daytime <= 1140:
                    self.assertTrue(world.weather['sun'] == 0,
                                    "Sun should not be shining now.")

    def test_wind_power_in_different_stapes(self):
        for world in self.worlds:
            for _ in range(1000):
                world.step()

                self.assertTrue(0 <= world.weather['wind'] <= 1,
                                "Wind power has incorrect value.")

    def test_clouds_interact_with_wind(self):
        for world in self.worlds:
            for _ in range(1000):
                world.step()

                if world.weather['wind']:
                    self.assertTrue(0 <= world.weather['clouds'] < 0.6,
                                    "Clouds value calculated with wind\
                                    is incorrect.")

    def test_clouds_interact_without_wind(self):
        for world in self.worlds:
            for _ in range(1000):
                world.step()

                if world.weather['wind'] == 0:
                    self.assertTrue(0 <= world.weather['clouds'] <= 0.6,
                                    "Clouds value calculated without wind\
                                    is incorrect.")

    def test_rain_not_falling(self):
        for world in self.worlds:
            for value in self.temp_for_rain_not_falling:
                world.weather['clouds'] = value
                world._calculate_rain()
                self.assertEqual(world.weather['rain'], 0,
                                 "Incorrect rain value - rain should not be\
                                 falling.")

    def test_rain_falling(self):
        for world in self.worlds:
            for value in self.temp_for_rain_falling:
                world.weather['clouds'] = value
                world._calculate_rain()
                self.assertEqual(world.weather['rain'], 1,
                                 "Incorrect rain value - rain should be\
                                 falling.")

    def test_light_value(self):
        for world in self.worlds:
            world.weather['sun'] = 0.4
            world.weather['clouds'] = 0.5
            world._calculate_light()
            self.assertEqual(world.weather['light'], 0.2,
                             "Incorrect light value.")

    def test_temperature_value(self):
        for world in self.worlds:
            for _ in range(1000):
                world.step()

                self.assertTrue(6.5 <= world.weather['temp'] <= 30.5,
                                "Clouds value calculated without wind\
                                is incorrect.")

    def test_weather_weight_scale(self):
        for world in self.worlds:
            self.assertEqual(world.previous_weather_weight
                             + world.current_weather_weight, 1,
                             "Incorrect weather weight scale values.")


if __name__ == '__main__':
    unittest.main()
