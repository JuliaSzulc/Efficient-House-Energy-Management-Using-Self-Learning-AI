from world import World
from house import House
from sensor_out import OutsideSensor
import re

class HouseEnergyEnvironment:
    """Endpoints / facade for RL environment.

    This is where we gather together our World, OutsideSensors, House, 
    etc., connect each other in a proper way and basically set up a working RL
    environment

    """

    def __init__(self):
        # functionality moved to reset() method, to be able to reinitialize the
        # whole environment
        self.reset()

    def step(self, action_name):
        """Step the environment by one timestep.
        
        Args:
            action_name(string): a name of action. For possible action names
                                 check get_actions() method
        Returns:
            observation(type?): information about the environment.
            reward(type?): a reward for RL agent's actions.
            done(boolean): information if the simulation has finished.

        """
        # TODO: type of observation, reward

        # make an action in the house
        getattr(self.house, action_name)()

        # move world forward
        self.world.step()

        # TODO: collect observations, calculate reward 
        observation, reward = None, None
        return observation, reward

    def reset(self):
        """(Re)initializes the environment"""

        self.world = World()
        self.outside_sensors = [OutsideSensor() for _ in range(1)]
        self.house = House()

        # register listeners:
        self.world.register(self.house)
        for outside_sensor in self.outside_sensors:
            self.world.register(outside_sensor)

        # TODO: other environment parts

    def render(self):
        """Outputs the state of environment in a human-readable format"""
        # TODO: print the state of environment to console. maybe simple
        # gui in the future?

    def get_actions(self):
        """Names of actions for RL-agent
        
        Returns:
            actions (list of strings): A list of method names
        
        Example:
            H = HouseEnergyEnvironment()
            actions = H.get_actions()

            # to make an action use pass its name to the step method, f.e.:
            H.step(actions[-1])

        """

        return [action for action in dir(self.house) 
                if callable(getattr(self.house, action))
                and re.match("action*", action)] 

