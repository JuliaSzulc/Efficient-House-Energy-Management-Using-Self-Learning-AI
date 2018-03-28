from world import World
from house import House
from sensor_out import OutsideSensor
import numpy as np
import re
import sys


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
            observation(dict): information about the environment. Consists of
                               'outside' and 'inside' dictionaries.
            reward(float): a reward for RL agent's action in the timeframe.
            done(boolean): information if the state after the step is terminal
                           (episode end).

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
        """Returns 1-dim ndarray of state parameters from dict

        Current array structure:

        [0] daytime //OUTSIDE
        [1] wind_chill
        [2] light
        [.] illumination
        [.] clouds
        [.] rain
        [ ] wind
        [ ] temperature //INSIDE
        [ ] light
        [ ] temp_desired
        [ ] temp_epsilon
        [ ] light_desired
        [ ] light_epsilon
        [ ] temp_desired
        [ ] temp_epsilon
        [ ] light_desired
        [ ] light_epsilon
        [ ] grid_cost
        [ ] battery_level

        """

        observation = []

        for sensor in state['outside']:
            for key, value in sensor.items():
                if key == 'wind_chill':
                    # temperature in range (-20, +40)'C
                    value = (value + 20) / 60
                observation.append(value)

        # NOTE: inconsistency - we have LIST of outside sensors and DICT of
        # inside sensors.

        for dk, dv in state['inside'].items():
            if dk == 'inside_sensors':
                for sensor in dv.values():
                    for key, value in sensor.items():
                        if re.match('temp.*', key):
                            # temperature in range (-20, +40)'C
                            value = (value + 20) / 60
                        observation.append(value)
            elif dk == 'desired':
                for daytime in dv.values():
                    for key, value in daytime.items():
                        if re.match('temp.*', key):
                            # temperature in range (-20, +40)'C
                            value = (value + 20) / 60
                        observation.append(value)
            else:
                observation.append(dv)

        # final safety zone = truncating everything
        def truncate(x):
            if not x: return 0
            if x < 0: return 0
            if x > 1: return 1
            return x

        # safely print message to error stream, but continue execution
        if not all([x and (0 <= x <= 1) for x in observation]):
            print("Whoa, some of observation values are not truncated to 0-1 or are None!",file=sys.stderr)

        observation = [truncate(x) for x in observation]
        return np.array([observation])
