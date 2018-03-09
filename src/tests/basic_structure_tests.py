import unittest
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from world import World
from listener import Listener

class BasicStructureTestCase(unittest.TestCase):
    """Testing observer/listener model and basic classes structure"""
    
    def setUp(self):
        self.world = World()
        self.listeners = [Listener(self.world) for _ in range(10)]
    
    def test_listeners_daytime(self):
        for step in range(10):
            self.world.step()
            for listener in self.listeners:
                self.assertEqual(self.world.daytime, listener.daytime)

if __name__ == "__main__":
    unittest.main()

