import unittest
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from world import World
from listener import Listener
from sensor_out import OutsideSensor
from environment import HouseEnergyEnvironment 

# TODO: testujac inne rzeczy niż strukturę klas (np. metody w danej klasie)
# utwórzcie nowy plik na wzór tego. Testując strukturę, ale w wyraźnie inny 
# sposób, utwórzcie nową klasę dziedziczącą po unittest.TestCase, a w niej 
# metodę setUp która przygotowuje obiekty i kolejno metody testujące. 

class BasicSubjectListenerTestCase(unittest.TestCase):
    """Testing observer/listener model and basic classes structure"""
    
    def setUp(self):
        self.world = World()
        self.listeners = [Listener(self.world) for _ in range(3)]
        self.sensors_out = [OutsideSensor(self.world) for _ in range(3)] 

    def test_all_listeners_daytime(self):
        for step in range(3):
            self.world.step()
            for listener in self.listeners:
                self.assertEqual(self.world.daytime, listener.daytime)
            
            for sensor_out in self.sensors_out:
                self.assertEqual(self.world.daytime, sensor_out.daytime)
 
    def test_all_listeners_weather(self):
        for step in range(3):
            self.world.step()
            for listener in self.listeners:
                for key in listener.weather.keys():
                    self.assertEqual(self.world.weather[key],\
                                     listener.weather[key])
                    
            for sensor_out in self.sensors_out:
                for key in sensor_out.weather.keys():
                    self.assertEqual(self.world.weather[key],\
                                     sensor_out.weather[key])


class EnvironmentStructureTestCase(unittest.TestCase):
    """Testing all environment relations"""

    def setUp(self):
        self.env = HouseEnergyEnvironment()

    def test_listeners(self):
        # TODO: make test
        pass
    
if __name__ == "__main__":
    unittest.main()

