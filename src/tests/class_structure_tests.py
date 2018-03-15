import unittest
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from world import World
from sensor_out import OutsideSensor
from environment import HouseEnergyEnvironment 
from house import House

# TODO: testujac inne rzeczy niż strukturę klas (np. metody w danej klasie)
# utwórzcie nowy plik na wzór tego. Testując strukturę klas, ale w wyraźnie inny 
# sposób, utwórzcie nową klasę dziedziczącą po unittest.TestCase, a w niej 
# metodę setUp która przygotowuje obiekty i kolejno metody testujące. 


class BasicSubjectListenerTestCase(unittest.TestCase):
    """Testing observer/listener model and basic classes structure"""
    
    def setUp(self):
        self.world = World()
        self.sensors_out = [OutsideSensor() for _ in range(3)] 
        self.house = House()
        self.world.register(self.house)
        for s in self.sensors_out:
            self.world.register(s)

    def test_all_listeners_daytime(self):
        for step in range(3):
            self.world.step()
            
            for sensor_out in self.sensors_out:
                self.assertEqual(
                    self.world.daytime,
                    sensor_out.daytime, 
                    "sensor out failed - wrong daytime"
                )

            self.assertEqual(
                self.world.daytime,
                self.house.daytime,
                "house failed - wrond daytime"
            )
 
    def test_all_listeners_weather(self):
        for step in range(3):
            self.world.step()
                    
            for sensor_out in self.sensors_out:
                for key in sensor_out.weather.keys():
                    self.assertEqual(
                        self.world.weather[key],
                        sensor_out.weather[key], 
                        "sensor out failed - wrong weather"
                    )

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

    def test_me(self):
        # TODO: make a test
        pass


if __name__ == "__main__":
    unittest.main()
