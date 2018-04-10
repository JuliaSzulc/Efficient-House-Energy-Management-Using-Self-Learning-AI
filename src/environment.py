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

        getattr(self.house, action_name)()
        done = self.world.step()
        observation = self._serialize_state(self._get_current_state())
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

        return self._serialize_state(self._get_current_state())

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
        # FIXME Args in docs are parameters the function receives,
        # FIXME not declares inside it
        # FIXME - render unnormalized values!

        dataset = self._serialize_state(self._get_current_state())
        reward = self.house.reward()
        dataset = np.append(dataset, reward)

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

        # ------------------ Removing ------------------
        # To remove a parameter, comment it out of the dict.
        # Then remove its value from dataSet using
        # dataSet = np.delete(dataSet, 0)
        # Be careful with indexes if you delete more than one parameter:
        # dataSet = np.delete(dataSet, 0)
        # dataSet = np.delete(dataSet, 3) # 4 is now 3

        return labels_names, dataset

    def get_actions(self):
        """Returns list of method names of possible actions

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
        return observation

    @staticmethod
    def _serialize_state(state):
        """Returns 1-dim ndarray of normalized state parameters from dict

        Args:
            state(dict) - the exact product of _get_current_state method.
            Note: Method assumes all temperature indicators are from range
            (-20, 40) C

        Returns(ndarray):
            Current array structure:

        [0] daytime // from Outside Sensor
        [1] temperature_outside
        [2] light
        [.] clouds
        [.] rain
        [ ] wind
        [ ] temperature // from House
        [ ] temperature_delta
        [ ] light
        [ ] temp_desired
        [ ] temp_epsilon
        [ ] light_desired
        [ ] light_epsilon
        [ ] energy_src
        [ ] cooling_lvl
        [ ] heating_lvl
        [ ] light_lvl
        [ ] curtains_lvl
        [ ] battery_level
        [ ] battery_delta
        """

        observation = []
        for sensor in state['outside']:
            for key, value in sensor.items():
                if key == 'actual_temp':
                    value = (value + 20) / 60
                elif key == 'daytime':
                    # time in range (0, 1440) min
                    value /= 1440
                elif key == 'illumination':
                    continue
                observation.append(value)

        for d_key, d_value in state['inside'].items():
            if d_key == 'inside_sensors':
                for sensor in d_value.values():
                    for key, value in sensor.items():
                        if re.match('temp.*', key):
                            value = (value + 20) / 60
                        observation.append(value)

            elif d_key == 'desired':
                for key, value in d_value.items():
                    if re.match('temp.*', key):
                        value = (value + 20) / 60
                    observation.append(value)

            elif d_key == 'devices_settings':
                for setting_key, setting_value in d_value.items():
                    if setting_key == 'energy_src':
                        setting_value = 1 if setting_value is 'grid' else 0
                    observation.append(setting_value)

            elif d_key == 'battery_level':
                d_value /= 14000
                observation.append(d_value)

            elif d_key == 'battery_delta':
                # FIXME better normalization of delta
                d_value /= 14000
                observation.append(d_value)
            else:
                observation.append(d_value)

        # FIXME move the assert below to tests?
        # make sure that vector is normalized. no safety zone - it has to work!
        assert all([x is not None and (0 <= x <= 1) for x in observation]), \
            "Whoa, some of observation values are not" + \
            "truncated to 0-1 or are None!" + \
            "vector: " + str(observation)

        return np.array(observation)
