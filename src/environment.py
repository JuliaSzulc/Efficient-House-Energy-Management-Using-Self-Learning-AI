"""This module provides environment for the RL model,

which is given in form of a class, based on examples from OpenAI
repositories. That class puts up together different environment elements,
and provides a nice facade for a model.

"""
import re
from collections import OrderedDict
import numpy as np
from world import World
from house import House
from sensor_out import OutsideSensor

def truncate(arg, lower=0, upper=1):
    """This function returns value truncated within range <lower, upper>

    Args:
        arg (number) - value to be truncated
        lower (number) - lower truncating bound, default to 0
        upper (number) - upper truncating bound, default to 1

    Returns:
        arg (number) - truncated function argument

    """

    if arg > upper:
        return upper
    if arg < lower:
        return lower
    return arg

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

        # transfer initial informations to listeners
        self.world.update_listeners()

        return self._get_current_state()

    def render(self):
        """Outputs the state of environment in a human-readable format

        Method takes numpy array returned by _get_current_astate() method.

        Args:
            dataset (numpy array) - its collection of values such as daytime,
            temperature, light etc. and aso reward

        Returns:
            labels_names(list) - names of specified values in dataSet
            dataSet (numpy array) - values of daytime, light etc.
        """
        # FIXME - render unnormalized values!
        # - gui in the future?
        # - maybe but in main, here just formatted text

        dataSet = self._get_current_state()

        reward = self.house.reward()

        dataSet = np.append(dataSet, reward)

        labels_names = ['Daytime //OUTSIDE: ',
                        'Temperature_outside: ',
                        'Light OUT: ',
                        'Illumination: ',
                        'Clouds: ',
                        'Rain: ',
                        'Wind: ',
                        'Temperature //INSIDE: ',
                        'Light IN: ',
                        'Temp_desired /current: ',
                        'Temp_epsilon: ',
                        'Light_desired: ',
                        'Light_epsilon: ',
                        'Grid_cost: ',
                        'Battery_level: ',
                        'TOTAL REWARD: '
                        ]

        # # ------------------ Removing tutorial ------------------
        # # to remove just comment the line, and remove its value from dataSet
        # labels_names = [#'Daytime //OUTSIDE: ',
        #                 'Temperature_outside: ',
        #                 'Light: ',
        #                 'Illumination: ',
        #                 #'Clouds: ',
        #                 #'Rain: ',
        #                 #'Wind: ',
        #                 'Temperature //INSIDE: ',
        #                 'Light: ',
        #                 'Temp_desired /current: ',
        #                 #'Temp_epsilon: ',
        #                 'Light_desired: ',
        #                 #'Light_epsilon: ',
        #                 'Grid_cost: ',
        #                 'Battery_level: ']
        #
        # # care for dynamic indexes change, if you remove element others
        # # indexes will change !!!
        # dataSet = np.delete(dataSet, 0)
        # dataSet = np.delete(dataSet, 3)
        # dataSet = np.delete(dataSet, 3)
        # dataSet = np.delete(dataSet, 3)
        # dataSet = np.delete(dataSet, 6)
        # dataSet = np.delete(dataSet, 7)
        #
        # # it will be error if dataSet length and labels_names length r not eq.
        # # --------------------------------------------------------------------

        return labels_names, dataSet

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
                and re.match("action.*", action)]

    def _get_current_state(self):
        outside_params = [sensor.get_info() for sensor in self.outside_sensors]
        inside_params = self.house.get_inside_params()
        observation = OrderedDict({
            'outside': outside_params,
            'inside': inside_params
        })
        return self._serialize_state(observation)

    @staticmethod
    def _serialize_state(state):
        """Returns 1-dim ndarray of state parameters from dict

        Current array structure:

        [0] daytime //OUTSIDE
        [1] temperature_outside
        [2] light
        [.] illumination
        [.] clouds
        [.] rain
        [ ] wind
        [ ] temperature //INSIDE
        [ ] light
        [ ] temp_desired / current
        [ ] temp_epsilon
        [ ] light_desired
        [ ] light_epsilon
        [ ] grid_cost
        [ ] battery_level
        FIXME where are deltas? (filip)
        FIXME throw out outside illumination - unnormalized light duplicate
        """

        observation = []
        # FIXME (filip) use get_current_user_requests
        time_of_day = 'night'  # to choose appropiate current desired temp
        for sensor in state['outside']:
            for key, value in sensor.items():
                if key == 'actual_temp':
                    # temperature in range (-20, +40)'C
                    value = (value + 20) / 60
                elif key == 'daytime':
                    # FIXME (filip) use get_current_user_requests
                    if 420 <= value <= 1080:
                        time_of_day = 'day'
                    # time in range (0, 1440) min
                    value /= 1440
                elif key == 'illumination':
                    # illumination in range (0, 25000)
                    value /= 25000
                observation.append(value)

        # NOTE: inconsistency - we have LIST of outside sensors and DICT of
        # inside sensors.

        for d_key, d_value in state['inside'].items():
            if d_key == 'inside_sensors':
                for sensor in d_value.values():
                    for key, value in sensor.items():
                        if re.match('temp.*', key):
                            # temperature in range (-20, +40)'C
                            value = (value + 20) / 60
                        observation.append(value)
            elif d_key == 'desired':
                for daytime, params in d_value.items():
                    # FIXME (filip) use get_current_user_requests
                    if daytime != time_of_day:
                        continue
                    for key, value in params.items():
                        if re.match('temp.*', key):
                            # temperature in range (-20, +40)'C
                            value = (value + 20) / 60
                        observation.append(value)
            elif d_key == 'battery_level':
                # battery in range (0, 14000) W
                d_value /= 14000
                observation.append(d_value)
            else:
                observation.append(d_value)

        # make sure that vector is normalized. no safety zone - it has to work!
        assert all([x is not None and (0 <= x <= 1) for x in observation]), \
            "Whoa, some of observation values are not" + \
            "truncated to 0-1 or are None!" + \
            "vector: " + str(observation)

        return np.array(observation)
