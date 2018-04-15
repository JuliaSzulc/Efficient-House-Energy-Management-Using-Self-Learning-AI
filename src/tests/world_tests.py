import unittest
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from world import World
from datetime import timedelta, datetime
import random


class TimestepsTestCase(unittest.TestCase):
    """Test if diferrent timesteps are giving us the same values"""

    def setUp(self):
        self.worlds = []
        timesteps = [0.5, 1, 5, 15, 30]
        # should work on random timesteps
        timesteps += [random.randrange(1, 100) for _ in range(10)]

        for timestep in timesteps:
            self.worlds.append(World(timestep))

    def test_sun(self):
        """Test if sun is independent from timeframe"""

        # checking if for the same daytime, sun is set to the same level
        # on different timesteps
        results = dict()
        for step in range(1000):
            for world in self.worlds:
                world.step()
                daytime = world.daytime
                if daytime in results.keys():

                    self.assertEqual(
                        results[daytime],
                        world.weather['sun'],
                        "sun should be independent from timeframe"
                    )

                else:
                    results[daytime] = world.weather['sun']


class StepEdgeTestCase(unittest.TestCase):
    """Test BASE vs CURRENT mechanism on edge cases"""

    def setUp(self):
        self.world_morning = World(random.randrange(3, 6), duration_days=2)
        self.world_midnight = World(random.randrange(3, 6), duration_days=2)
        self.world_short = World(5)

        self.world_morning.current_date = datetime(2020, 1, 1, 0, 0, 0)
        self.world_morning.stop_date = datetime(2020, 1, 2, 0, 0, 0)
        self.world_morning._compute_daytime()
        self.world_morning._compute_basetime()

        self.world_midnight.current_date = datetime(2020, 1, 1, 23, 50, 0)
        self.world_midnight.stop_date = datetime(2020, 1, 2, 0, 10, 0)
        self.world_midnight._compute_daytime()
        self.world_midnight._compute_basetime()

        self.world_short.current_date = datetime(2020, 1, 1, 0, 0, 0)
        self.world_short.stop_date = datetime(2020, 1, 1, 0, 3, 0)
        self.world_short._compute_daytime()
        self.world_short._compute_basetime()

    def test_midnight(self):
        """Should work when base date is 00:00:00 before current date"""
        while not self.world_midnight.step():
            current_date = self.world_midnight.current_date
            base_date = self.world_midnight.base_date
            self.assertTrue(current_date <= base_date)

    def test_morning(self):
        """Should work when current date is 00:00:00 along with base date"""
        while not self.world_morning.step():
            current_date = self.world_morning.current_date
            base_date = self.world_morning.base_date
            self.assertTrue(current_date <= base_date)

    def test_no_action_after_stop_date(self):
        """There should be no action after stop date"""
        while not self.world_short.step():
            pass
        weather = dict(self.world_short.weather)
        interpolated = dict(self.world_short.int_weather)
        daytime = self.world_short.daytime
        basetime = self.world_short.daytime

        for step in range(10):
            self.world_short.step()

        self.assertEqual(weather, self.world_short.weather,
                         "weather should not be changed after stop date")
        self.assertEqual(interpolated, self.world_short.int_weather,
                         "interp. weather shouldnt be chnged after stop date")
        self.assertEqual(daytime, self.world_short.daytime,
                         "daytime should not be changed after stop date")
        self.assertEqual(basetime, self.world_short.basetime,
                         "basetime should not be changed after stop date")


class DurationTestCase(unittest.TestCase):
    """Test if diferrent episode duration has no influence"""

    def setUp(self):
        self.worlds = []
        duration_list = [0.5, 1, 1.213, 2, 5]

        for duration in duration_list:
            self.worlds.append(World(duration_days=duration))

    def test_sun(self):
        """Test if sun is independent from duration"""

        # checking if for the same daytime, sun is set to the same level
        # on different timesteps
        results = dict()
        finished = set()
        while len(finished) != len(self.worlds):
            for index, world in enumerate(self.worlds):
                if index in finished:
                    continue
                finish = world.step()
                if finish:
                    finished.add(index)
                    continue

                daytime = world.daytime
                if daytime in results.keys():

                    self.assertEqual(
                        results[daytime],
                        world.weather['sun'],
                        "sun should be independent from duration"
                    )

                else:
                    results[daytime] = world.weather['sun']


class WeatherTestCase(unittest.TestCase):
    """Test different weather computations"""
    # NOTE: there is no point in unittest algorithms based on propabilities.
    # Use plot function from world.py to check if current algorithms suit your
    # needs.

    def setUp(self):
        self.world = World()

    def test_sun(self):
        """Sun should shine in appropiate range of daytime"""
        daystart = 300
        dayend = 1140
        while not self.world.step():
            if dayend > self.world.daytime > daystart:
                self.assertTrue(self.world.weather['sun'] != 0,
                                "sun should shine now")

    def test_temperature_range(self):
        """Temperature should always be in appropiate range"""
        min = -20
        max = 40
        while not self.world.step():
            self.assertTrue(min <= self.world.weather['sun'] <= max,
                            "temperature is out of range")


if __name__ == '__main__':
    unittest.main()
