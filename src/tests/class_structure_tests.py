import unittest
from src.world import World
from src.sensor_out import OutsideSensor
from src.environment import HouseEnergyEnvironment
from src.house import House

# TODO: testujac inne rzeczy niż strukturę klas (np. metody w danej klasie)
# utwórzcie nowy plik na wzór tego. Testując strukturę klas, ale w wyraźnie inny
# sposób, utwórzcie nową klasę dziedziczącą po unittest.TestCase, a w niej
# metodę setUp która przygotowuje obiekty i kolejno metody testujące.


class BasicSubjectListenerTestCase(unittest.TestCase):
    """Testing observer/listener model and basic classes structure"""

    def setUp(self):
        self.world = World()
        self.sensors_out = [OutsideSensor() for _ in range(3)]
        self.house = House(self.world.time_step_in_minutes)
        self.world.register(self.house)
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
                "house failed - wrond daytime"
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

    def test_house_weather(self):
        """Test house weather vs world weather"""
        for step in range(3):
            self.world.step()

            for key in self.world.weather.keys():
                self.assertEqual(
                    self.world.weather[key],
                    self.house.weather[key],
                    "house failed - wrond weather"
                )


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
