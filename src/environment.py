from src.world import World
from src.house import House
from src.sensor_out import OutsideSensor
import numpy as np
import re


class HouseEnergyEnvironment:
    """Endpoints / facade for RL environment.

    This is where we gather together our World, OutsideSensors, House, 
    etc., connect each other in a proper way and basically set up a working RL
    environment

    """

    def __init__(self):
        # Remember to put all the declarations of class fields here!

        self.world = None
        self.outside_sensors = None
        self.house = None

        # Actual initialization is moved to reset() method,
        # to be able to re-initialize the whole environment.
        self.reset()

    def step(self, action_name):
        """Step the environment by one timestep.
        
        Args:
            action_name(string): a name of action. For possible action names
                                 check get_actions() method
        Returns:
            observation(dict): information about the environment. Consists of 'outside' and 'inside' dictionaries.
            reward(float): a reward for RL agent's action in the timeframe.
            done(boolean): information if the state after the step is terminal (episode end).

        """
        # make an action in the house
        getattr(self.house, action_name)()

        # make the step in the world.
        done = self.world.step()

        # get new environment state, calculate reward
        observation = self._get_current_state()
        reward = self.house.reward()
        return observation, reward, done

    def reset(self):
        """(Re)initializes the environment

        Returns:
            Initial state of the environment
        """

        self.world = World()
        self.house = House(self.world.time_step_in_minutes)
        self.outside_sensors = [OutsideSensor(self.house) for _ in range(1)]

        # register listeners:
        for outside_sensor in self.outside_sensors:
            self.world.register(outside_sensor)

        # TODO: other environment parts
        return self._get_current_state()

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

    def _get_current_state(self):
        outside_params = [sensor.get_info() for sensor in self.outside_sensors]
        inside_params = self.house.get_inside_params()
        observation = {'outside': outside_params, 'inside': inside_params}
        return self._serialize_state(observation)

    def _serialize_state(self, state):
        """Returns 1-dim ndarray of state parameters from dict"""
        # TODO implement me!
        return np.array([])
