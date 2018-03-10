from world import World
from sensor_out import OutsideSensor


class HouseEnergyEnvironment:
    """Endpoints / facade for RL environment.

    This is where we gather together our World, OutsideSensors, InsideSensors, 
    etc., connect each other in a proper way and basically set up a working RL
    environment

    """

    def __init__(self):
        # functionality moved to reset() method, to be able to reinitialize the
        # whole environment
        self.reset()

    def step(self, action):
        """Step the environment by one timestep.
        
        Args:
            action(type?): the action made by RL agent. 
        Returns:
            observation(type?): information about the environment.
            reward(type?): a reward for RL agent's actions.
            done(boolean): information if the simulation has finished.

        """
        # TODO: type of action, observation, reward
        # TODO: do sth with action
        self.world.step()

        # TODO: collect observations, calculate reward 

        return observation, reward

    def reset(self):
        """(Re)initializes the environment"""

        self.world = World()
        self.outside_sensors = [OutsideSensor(self.world) for _ in range(1)]
        # TODO: other environment parts

    def render(self):
        """Outputs the state of environment in a human-readable format"""
        # TODO: print the state of environment to console. maybe simple
        # gui in the future?
