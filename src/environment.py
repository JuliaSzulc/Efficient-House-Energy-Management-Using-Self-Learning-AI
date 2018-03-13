from world import World
from house import House
from sensor_out import OutsideSensor


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

    def step(self):
        """Step the environment by one timestep.
        
        Returns:
            observation(type?): information about the environment.
            reward(type?): a reward for RL agent's actions.
            done(boolean): information if the simulation has finished.

        """
        # TODO: type of observation, reward
        # NOTE: action will not be passed, but simply executed 
        # in another way (ask Filip)
        self.world.step()

        # TODO: collect observations, calculate reward 
        observation, reward = None, None
        return observation, reward

    def reset(self):
        """(Re)initializes the environment"""

        self.world = World()
        self.outside_sensors = [OutsideSensor(self.world) for _ in range(1)]
        self.house = House(self.world)
        # TODO: other environment parts

    def render(self):
        """Outputs the state of environment in a human-readable format"""
        # TODO: print the state of environment to console. maybe simple
        # gui in the future?

    def get_actions(self):
        """Ready to use functions for RL-agent
        
        Returns:
            actions (list of functions): A list of methods
            house: A "self" reference, to be passed as first argument
        
        Example of use:
            H = HouseEnergyEnvironment()
            actions, param = H.get_actions()

            # to make an action:
            actions[i](param) # ...and that's it :)

        """

        return self.house.actions, self.house

