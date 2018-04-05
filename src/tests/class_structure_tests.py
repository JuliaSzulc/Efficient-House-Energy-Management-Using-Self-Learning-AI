import unittest
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from world import World
from sensor_out import OutsideSensor
from environment import HouseEnergyEnvironment
from house import House


class BasicSubjectListenerTestCase(unittest.TestCase):
    """Testing observer/listener model and basic classes structure"""

    def setUp(self):
        self.world = World()
        self.house = House(self.world.time_step_in_minutes)
        self.sensors_out = [OutsideSensor(self.house) for _ in range(3)]
        for s in self.sensors_out:
            self.world.register(s)

    def test_sensors_daytime(self):
        """Test sensors daytime vs world daytime"""
        for step in range(3):
            self.world.step()

            for sensor_out in self.sensors_out:
                self.assertEqual(
                    self.world.daytime,
                    sensor_out.daytime,
                    "sensor out failed - wrong daytime"
                )

    def test_house_daytime(self):
        """Test house daytime vs world daytime"""
        for step in range(3):
            self.world.step()

            self.assertEqual(
                self.world.daytime,
                self.house.daytime,
                "house failed - wrong daytime"
            )

    def test_sensors_weather(self):
        """Test house weather vs world weather"""
        for step in range(3):
            self.world.step()

            for sensor_out in self.sensors_out:
                for key in sensor_out.weather.keys():
                    self.assertEqual(
                        self.world.weather[key],
                        sensor_out.weather[key],
                        "sensor out failed - wrong weather"
                    )

    def test_catch_attrib_error(self):
        """Testing listener with unimplemented method update"""

        listener = []
        self.world.register(listener)
        with self.assertRaises(AttributeError):
            self.world.update_listeners()


class EnvironmentStructureTestCase(unittest.TestCase):
    """Testing all environment relations"""

    def setUp(self):
        self.env = HouseEnergyEnvironment()

    def test_ownership_in_environment(self):
        """Testing if environment has all necessary things"""

        self.assertIn("house", dir(self.env))
        self.assertIn("world", dir(self.env))
        self.assertIn("outside_sensors", dir(self.env))

    def test_attributes_initialized(self):
        """Testing if env attributes are initialized and not None"""

        self.assertIsNotNone(self.env.house)
        self.assertIsNotNone(self.env.world)
        self.assertIsNotNone(self.env.outside_sensors)


if __name__ == "__main__":
    unittest.main()
