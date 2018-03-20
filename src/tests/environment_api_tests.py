import unittest
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from environment import HouseEnergyEnvironment
import types


class BasicEnvironmentTestCase(unittest.TestCase):
    """Testing basic mocked environment usage"""

    def setUp(self):
        self.env = HouseEnergyEnvironment()

        def action(self):
            self.mocked_param += 1

        self.env.house.mocked_param = 0
        self.env.house.action_mocked = types.MethodType(action, self.env.house)

    def test_get_action(self):
        """test getting mocked action"""

        actions = self.env.get_actions()
        self.assertTrue(
            "action_mocked" in actions,
            "failed at getting actions!"
        )

    def test_make_action_in_step(self):
        """test making mocked action"""
        self.env.step("action_mocked")

        self.assertEqual(
            self.env.house.mocked_param,
            1,
            "mocked action on environment failed!"
        )


if __name__ == '__main__':
    unittest.main()
