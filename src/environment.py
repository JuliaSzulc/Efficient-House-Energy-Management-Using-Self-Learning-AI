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

    def __init__(self, world=None):
        """Actual initialization is moved to reset() method

        to be able to re-initialize the whole environment.

        """

        self.world = None
        self.outside_sensors = None
        self.house = None

        self.last_reward = 0

        self.timesteps = 0
        self.temp_diff_2_count = 0
        self.temp_diff_05_count = 0
        self.light_diff_015_count = 0
        self.light_diff_005_count = 0

        self.reset(world)

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
        current_state = self._get_current_state()
        observation = self._serialize_state(current_state)

        self.last_reward = self.house.reward()
        self._update_stats(current_state['inside'])

        return observation, self.last_reward, done

    def reset(self, world=None):
        """(Re)initializes the environment

        Returns:
            Initial state of the environment
        """

        self.world = world or World()

        self.timesteps = 0
        self.temp_diff_2_count = 0
        self.temp_diff_05_count = 0
        self.light_diff_015_count = 0
        self.light_diff_005_count = 0

        self.house = House(self.world.time_step_in_minutes)
        self.outside_sensors = [OutsideSensor(self.house) for _ in range(1)]

        # register listeners:
        for outside_sensor in self.outside_sensors:
            self.world.register(outside_sensor)

        # transfer initial informations to listeners
        self.world.update_listeners()

        return self._serialize_state(self._get_current_state())

    @property
    def render(self):
        """Outputs the state of environment in a human-readable format

        Returns:
            labels(list) - names for each value in data
            data(numpy array) - values of environment parameters
        """

        reward = self.last_reward

        # --- unnormalized ---
        unnormalized_dataset = []
        d = self._get_current_state()

        for sensor in d['outside']:
            for key, value in sensor.items():
                unnormalized_dataset.append(value)

        for d_key, d_value in d['inside'].items():
            if d_key == 'inside_sensors':
                for sensor in d_value.values():
                    for key, value in sensor.items():
                        unnormalized_dataset.append(value)
            elif d_key == 'desired' or d_key == 'devices_settings':
                for key, value in d_value.items():
                    unnormalized_dataset.append(value)
            else:
                unnormalized_dataset.append(d_value)
        unnormalized_dataset.append(reward)

        # --- normalized ---
        dataset = self._serialize_state(self._get_current_state())
        dataset = np.append(dataset, reward)

        # --- tags ---
        labels_names = [
            'Daytime //OUTSIDE: ',
            'Temperature_outside: ',
            'Light OUT: ',
            'Clouds: ',
            'Rain: ',
            'Wind: ',
            'Temperature //INSIDE: ',
            'Temperature_delta: ',
            'Light IN: ',
            'Temp_desired: ',
            'Temp_epsilon: ',
            'Light_desired: ',
            'Light_epsilon: ',
            'Grid_cost: ',
            'Energy_src: ',
            'Cooling_lvl: ',
            'Heating_lvl: ',
            'Light_lvl: ',
            'Curtains_lvl: ',
            'Battery_lvl: ',
            'Battery_delta: ',
            'TOTAL REWARD: '
        ]

        return labels_names, unnormalized_dataset, dataset

    def get_actions(self):
        """Returns list of method names (possible actions)

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
        [ ] grid_cost
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
                d_value /= 10
                observation.append(d_value)
            else:
                observation.append(d_value)

        # NOTE move the assert below to tests, (in the future)
        # make sure that vector is normalized. no safety zone - it has to work!
        assert all([x is not None and (0 <= x <= 1) for x in observation]), \
            "Whoa, some of observation values are not" + \
            "truncated to 0-1 or are None!" + \
            "vector: " + str(observation)

        return np.array(observation)

    def get_episode_stats(self):

        temp_2 = 100 * self.temp_diff_2_count / self.timesteps
        temp_05 = 100 * self.temp_diff_05_count / self.timesteps
        light_015 = 100 * self.light_diff_015_count / self.timesteps
        light_005 = 100 * self.light_diff_005_count / self.timesteps

        return {'Temperature difference < 2': temp_2,
                'Temperature difference < 0.5': temp_05,
                'Light difference < 0.15': light_015,
                'Light difference < 0.05': light_005}

    def _update_stats(self, state):
        self.timesteps += 1

        temp_difference = abs(state['inside_sensors']['first']['temperature']
                              - state['desired']['temp_desired'])

        light_difference = abs(state['inside_sensors']['first']['light']
                               - state['desired']['light_desired'])

        if temp_difference < 2:
            self.temp_diff_2_count += 1
            if temp_difference < 0.5:
                self.temp_diff_05_count += 1
        if light_difference < 0.15:
            self.light_diff_015_count += 1
            if light_difference < 0.05:
                self.light_diff_005_count += 1
