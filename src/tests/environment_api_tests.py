import unittest
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from environment import HouseEnergyEnvironment

class BasicEnvironmentTestCase(unittest.TestCase):
    """Testing basic mocked environment usage"""

    def setUp(self):
        self.env = HouseEnergyEnvironment()
        
        def action(self, param):
            self.mocked_param += param

        self.env.house.mocked_param = 0
        self.env.house.mocked_action = action 
        self.env.house.actions.append(action)

    def test_action(self):
        actions, ref = self.env.get_actions()
        actions[-1](ref, 5) # make an action

        self.assertEqual(
            self.env.house.mocked_param,
            5,
            "mocked action on environment failed!"
        )

if __name__ == '__main__':
    unittest.main()
