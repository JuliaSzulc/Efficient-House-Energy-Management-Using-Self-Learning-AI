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
    environment.

    """

    def __init__(self, world=None, collect_stats=False):
        """
        Declares all class' fields.

        Initialization is moved to reset() method
        to be able to quickly re-initialize the whole environment.

        Stats-related fields describe how many times given parameter
        was close to the desired value within a particular interval
        in current episode (see _update_stats method)

        Args:
            world(World): (optional) a World object to be used in environment
            collect_stats(boolean): (optional)

        """

        self.world = None
        self.outside_sensors = None
        self.house = None

        self.last_reward = 0

        self.collect_stats = collect_stats
        self.timesteps = 0
        self.temp_diff_2_count = 0
        self.temp_diff_05_count = 0
        self.light_diff_015_count = 0
        self.light_diff_005_count = 0

        self.reset(world)

    def step(self, action_name):
        """
        Update the environment by one timestep and update the statistics.
        This method is the main communication point between
        agent and the environment.

        Args:
            action_name(string): a name of action. For possible action names
                                 check get_actions() method
        Returns:
            observation(dict): serialized information about the environment
            reward(float): a reward for RL agent's last action
            done(boolean): information whether the new state, achieved after
                           this update, is terminal (episode end)

        """

        getattr(self.house, action_name)()
        done = self.world.step()
        current_state = self.get_current_state()

        if self.collect_stats:
            self._update_stats(current_state)

        observation = self.serialize_state(current_state)
        self.last_reward = self.house.reward()

        return observation, self.last_reward, done

    def reset(self, world=None):
        """(Re)initializes the environment and registers the listeners.

        Should be used to start a new episode. Returns the first,
        serialized initial state.

        Returns:
            Serialized initial state of the environment
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

        # transfer initial information to listeners
        self.world.update_listeners()

        return self.serialize_state(self.get_current_state())

    def get_actions(self):
        """Returns list of action-method names (possible actions)

        Returns:
            actions (list of strings): A list of action-method names

        Example:
            H = HouseEnergyEnvironment()
            actions = H.get_actions()

            # to make an action use pass its name to the step method, f.e.:
            H.step(actions[-1])

        """

        return [action for action in dir(self.house)
                if callable(getattr(self.house, action))
                and re.match("action.*", action)]

    def get_current_state(self):
        """Returns a dicitonary of unnormalized environment state values.

        Use this method to gather human-readable information about the state.
        This method shouldn't be used by the reinforcement learning agent, as
        the values aren't normalized. Normalized state is returned by the step
        method, which uses _serialize_state() method to normalize this dict.

        Returns(OrderedDict):
            Current outside and inside sensor values, user
            desired values, action-controllable settings of devices,
            daytime and reward for the last timeframe.
        """
        outside_params = [sensor.get_info() for sensor in self.outside_sensors]
        inside_params = self.house.get_inside_params()

        current_state = OrderedDict([])

        for sensor_info in outside_params:
            current_state.update(sensor_info)

        for param_key, param_value in inside_params.items():
            if param_key == 'inside_sensors':
                for sensor_key, sensor_values in param_value.items():
                    for key, value in sensor_values.items():
                        current_state[sensor_key + "." + key] = value
            elif param_key == "desired" or param_key == "devices_settings":
                for key, value in param_value.items():
                    current_state[key] = value
            else:
                current_state[param_key] = param_value

        current_state['Reward'] = self.last_reward

        return current_state

    @staticmethod
    def serialize_state(state):
        """Returns 1-dim ndarray of normalized state parameters from dict

        Args:
            state(OrderedDict) - the exact product of _get_current_state method.

        Note: Method assumes all temperature indicators are from range
        (-20, 40) + every temperature indicator contains 'temp' in the key name
        - and this is a project global assumption.

        Notice that this method gets deletes the daytime entry and any constant
        values.

        Returns(ndarray):
            Normalized, 'neural-net ready' state 1-dim array.

        Current array structure:
        [0] Outside Temperature
        [1] Outside Light
        [2] Clouds
        [.] Rain
        [.] Wind
        [.] temperature
        [ ] temperature_delta
        [ ] light
        [ ] temp_desired
        [ ] light_desired
        [ ] grid_cost
        [ ] energy_src
        [ ] cooling_lvl
        [ ] heating_lvl
        [ ] light_lvl
        [ ] curtains_lvl
        [ ] battery_level
        [ ] battery_delta
        """

        del state['Daytime']
        del state['Reward']

        for key, value in state.items():
            if re.match('.*temp.*', key, re.IGNORECASE):
                state[key] = (value + 20) / 60

        state['energy_src'] = 1 if state['energy_src'] is 'grid' else 0
        state['battery_level'] /= state['battery_max']
        del state['battery_max']
        state['battery_delta'] /= 10

        return np.array(list(state.values()))

    def get_episode_stats(self):
        """Provides statistic info about episode.

        Returns:
            dictionary with current statistics expressed in percent of
            current episode time.

        Returns the correct values only if the environment works in the
        collect_stats mode and there was at least one step taken; returns None
        if not. The stats functionality works only for one sensor version
        of the environment
        """

        if self.collect_stats and self.timesteps != 0:
            temp_2 = 100 * self.temp_diff_2_count / self.timesteps
            temp_05 = 100 * self.temp_diff_05_count / self.timesteps
            light_015 = 100 * self.light_diff_015_count / self.timesteps
            light_005 = 100 * self.light_diff_005_count / self.timesteps

            return {'Temperature difference < 2': temp_2,
                    'Temperature difference < 0.5': temp_05,
                    'Light difference < 0.15': light_015,
                    'Light difference < 0.05': light_005}
        else:
            return None

    def _update_stats(self, state):
        """Updates the statistics of fulfilling the desired values.

        Updating stats is done by checking the absolute
        difference between current and desired values. Current values are
        taken from the first, main sensor in the house.

        If the difference is smaller than given value, the statistic is
        increased. Note that the statistics are just counts - the episode
        percents are calculated in the get_episode_stats method.

        Args:
            state(OrderedDict): dictionary in format returned by
                                _get_current_state() method
        """

        self.timesteps += 1
        temp_difference = abs(state['first.temperature']
                              - state['temp_desired'])
        light_difference = abs(state['first.light']
                               - state['light_desired'])

        if temp_difference < 2:
            # TODO: add test for this condition
            self.temp_diff_2_count += 1
            if temp_difference < 0.5:
                self.temp_diff_05_count += 1
        if light_difference < 0.15:
            # TODO: add test for this condition
            self.light_diff_015_count += 1
            if light_difference < 0.05:
                self.light_diff_005_count += 1
