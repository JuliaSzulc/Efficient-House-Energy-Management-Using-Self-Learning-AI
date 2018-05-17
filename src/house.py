"""This module provides the House utilities

in form of a House class and additional functions. House class is the most
important part of HouseEnergyEnvironment structure, in which most of the actual
actions take place. It simulates light and energy distribution and cost,
contains actions for RL agent to change inside parameters such as heating or
light levels; it also calculates the reward / penalty for the last timeframe.

Is is mainly used within HouseEnergyEnvironment class, and should not be used
directly from outside.

Note: Currently, the sensor has no specific impact on the registered values, so
every sensor would register the same values. There is only one, main sensor
used at the moment, and you have to

"""
import json
import random
from collections import OrderedDict
from tools import truncate


class House:
    """Main environment part"""

    def __init__(self, timeframe):
        with open('../configuration.json') as config_file:
            self.config = json.load(config_file)

        # Time constants and variables (in minutes)
        self.timeframe = timeframe
        self.day_start = 7 * 60
        self.day_end = 18 * 60
        self.daytime = 0

        # Energy and light settings
        self.pv_absorption = 5  # Watt/min on max sun intensity
        self.grid_cost = 0.5
        self.house_isolation_factor = 0.998
        self.house_light_factor = 0.0075
        self.max_led_illuminance = 200  # lux
        self.max_outside_illumination = 25000  # lux - max. in ambient daylight
        self.battery = {
            'current': 0,
            'delta': 0,
            'max': 14000  # Watt, as good as single Tesla PowerWall unit.
        }
        self.devices_power = {
            'air_conditioner': 1500,
            'heater': 3000,
            'light': 20
        }

        self.user_requests = OrderedDict({
                'temp_desired': 18,
                'temp_epsilon': 1,
                'light_desired': 0.0,
                'light_epsilon': 0.05
        })

        self.inside_sensors = OrderedDict({
            'first': OrderedDict({
                'temperature': 18,
                'temperature_delta': 0,
                'light': 0
            })
        })

        # Action-controlled settings
        self.devices_settings = OrderedDict({
            'energy_src': 'grid',  # grid/battery
            'cooling_lvl': 0,
            'heating_lvl': 0,
            'light_lvl': 0,
            'curtains_lvl': 0
        })

        self.illegal_action_penalty = 0

        # actions influence on current settings - default to 0.2 / min
        self.influence = 0.2 * timeframe

    def _update_grid_cost(self):
        """Updates the grid cost based on daytime. Expressed in PLN for 1kWh"""
        if self.day_start < self.daytime < self.day_end:
            self.grid_cost = 0.5
        else:
            self.grid_cost = 0.3

    def _update_user_requests(self):
        """
        Randomly updates the user requests every 2 hours

        For the temperature, requests are integers from <19, 24> interval during
        the day, and from <15, 20> interval during the night time.

        For the light, interval remain the same during the whole time - <0, 1>
        Values are rounded to 1 decimal point.
        """

        if self.daytime % 120 == 0:
            self.user_requests['temp_desired'] = random.randint(15, 20)

            if self.day_start < self.daytime < self.day_end:
                self.user_requests['temp_desired'] += 4

            self.user_requests['light_desired'] = round(random.random(), 1)

    def _calculate_light(self, outside_illuminance):
        """
        Updates the inside sensors with new light value. Final light is
        normalized to <0, 1> interval and depends on the
        light device level, curtains level, outside light level and
        house_light_factor, which describes how much illuminance 'enters'
        the house.

        Note: Currently, the sensor has no impact on the registered value, so
        every sensor would register the same value.
        """

        for data in self.inside_sensors.values():
            outside_light = (outside_illuminance * self.house_light_factor) \
                * (1 - self.devices_settings['curtains_lvl'])

            inside_light = self.devices_settings['light_lvl'] \
                * self.max_led_illuminance

            final_light = (outside_light + inside_light) \
                / self.max_led_illuminance

            data['light'] = truncate(final_light)

    def _calculate_temperature(self, outside_temp):
        """
        Updates the inside sensors with new temperature value. The new value
        depends on the last temperature, the outside temperature, house's
        isolation factor, and levels of heating and cooling devices.
        Value is truncated to <-20, 40> interval and this interval is assumed
        in the environment's normalization method -
        HouseEnergyEnvironment.serialize_state().

        Note: Currently, the sensor has no impact on the registered value, so
        every sensor would register the same value.
        """
        for data in self.inside_sensors.values():
            last_inside_temp = data['temperature']
            temp_delta = (outside_temp - last_inside_temp) \
                * (1 - self.house_isolation_factor)

            new_inside_temp = last_inside_temp \
                + self.timeframe \
                * (temp_delta + self.devices_settings['heating_lvl']
                    - self.devices_settings['cooling_lvl']) / 5

            data['temperature_delta'] = new_inside_temp - last_inside_temp
            data['temperature'] = truncate(new_inside_temp, -20, 40)

    def _calculate_accumulated_energy(self, outside_light):
        """Calculates new value of energy accumulated in the battery"""
        acc = outside_light * self.pv_absorption * self.timeframe
        self.battery['delta'] = acc
        self.battery['current'] = truncate(arg=(acc + self.battery['current']),
                                           upper=self.battery['max'])

    def update(self, sensor_out_info):
        """Updates house parameters

        Args:
            sensor_out_info(dict) - weather and time information from
                                     outside sensor
        """

        self.daytime = sensor_out_info['daytime']
        self._update_grid_cost()
        self._update_user_requests()
        self._calculate_accumulated_energy(sensor_out_info['light'])
        self._calculate_temperature(sensor_out_info['actual_temp'])
        self._calculate_light(sensor_out_info['light']
                              * self.max_outside_illumination)

    def get_inside_params(self):
        """Returns unnormalized information about the state of the house

        Returns:
            inside_params (OrderedDict): A dictionary with unnormalized
            house information

        Structure of returned dict consist of:
            'inside_sensors' - dict of inside sensors info
            'desired' - dict of settings requested by user
            'grid_cost' - a cost of energy
            'devices_settings' - levels of action-dependent settings
            'battery_level' - current battery level
            'battery_delta' - accumulation tempo of the battery
        """

        inside_params = OrderedDict({
            'inside_sensors': self.inside_sensors,
            'desired': self.user_requests,
            'grid_cost': self.grid_cost,
            'devices_settings': self.devices_settings,
            'battery_level': self.battery['current'],
            'battery_delta': self.battery['delta']
        })

        return inside_params

    def _calculate_device_energy_usage(self, device_full_power,
                                       device_setting):
        """
        Calculates cost of last time-frame's energy usage of given device

        Args:
            device_full_power(numeric): full device power in kWh
            device_setting(numeric): value from 0 to 1
        Returns:
            energy usage of last timeframe expressed in kWh
        """

        return device_full_power * device_setting * (self.timeframe / 60)

    def _calculate_energy_cost(self):
        """
        Calculates the cost of energy usage of last time-frame in the whole
        house. Energy used from photovoltaic battery is free. If the usage
        exceeded the battery level, the source of energy is switched to grid.
        Returns:
            cost of energy usage - usage multiplied by current grid_cost

        """
        usage = self._calculate_device_energy_usage(
            self.devices_power['air_conditioner'],
            self.devices_settings['cooling_lvl']
        )
        usage += self._calculate_device_energy_usage(
            self.devices_power['heater'],
            self.devices_settings['heating_lvl']
        )
        usage += self._calculate_device_energy_usage(
            self.devices_power['light'],
            self.devices_settings['light_lvl']
        )

        if self.devices_settings['energy_src'] == 'battery':
            if self.battery['current'] > usage:
                self.battery['current'] -= usage
                return 0
            else:
                # Used more energy than we had in battery
                usage -= self.battery['current']
                self.battery['current'] = 0
                self.devices_settings['energy_src'] = 'grid'

        return usage * self.grid_cost

    def reward(self):
        """
        Calculate reward for the last timeframe.
        Note that the reward in the whole simulator is always non-positive,
        so it is easier to interpret as penalty in this case.

        Function is parametrized by weights for every factor's penalty
        and by exponents (for temp and light penalties)
        To see how exponents are used, check _calculate_penalty() method

        Returns:
             reward(float): weighted sum of penalties
        """

        w_temp = self.config['env']['temperature_w_in_reward']
        w_light = self.config['env']['light_w_in_reward']
        w_cost = self.config['env']['cost_w_in_reward']

        cost = self._calculate_energy_cost()
        temp, light = (self.inside_sensors['first']['temperature'],
                       self.inside_sensors['first']['light'])
        req = self.user_requests

        temp_penalty = abs(temp - req['temp_desired'])

        light_penalty = abs(light - req['light_desired'])

        reward = -1 * ((cost * w_cost)
                       + (temp_penalty * w_temp)
                       + (light_penalty * w_light))

        return reward / 5 - (1 if self.illegal_action_penalty else 0)

    # All action-method names (and only them) have to start with "action"!

    def action_source_grid(self):
        """Action to be taken by RL-agent - change power source"""
        self.illegal_action_penalty = 1 if \
            self.devices_settings['energy_src'] == 'grid' else 0

        self.devices_settings['energy_src'] = 'grid'

    def action_source_battery(self):
        """Action to be taken by RL-agent - change power source"""
        self.illegal_action_penalty = 1 if \
            self.devices_settings['energy_src'] == 'battery' else 0

        self.devices_settings['energy_src'] = 'battery'

    def action_more_cooling(self):
        """Action to be taken by RL-agent"""
        self.illegal_action_penalty = 1 if \
            self.devices_settings['cooling_lvl'] == 1 else 0

        self.devices_settings['cooling_lvl'] = round(
            truncate(self.devices_settings['cooling_lvl'] + self.influence), 2)

    def action_less_cooling(self):
        """Action to be taken by RL-agent"""
        self.illegal_action_penalty = 1 if \
            self.devices_settings['cooling_lvl'] == 0 else 0

        self.devices_settings['cooling_lvl'] = round(
            truncate(self.devices_settings['cooling_lvl'] - self.influence), 2)

    def action_more_heating(self):
        """Action to be taken by RL-agent"""
        self.illegal_action_penalty = 1 if \
            self.devices_settings['heating_lvl'] == 1 else 0

        self.devices_settings['heating_lvl'] = round(
            truncate(self.devices_settings['heating_lvl'] + self.influence), 2)

    def action_less_heating(self):
        """Action to be taken by RL-agent"""
        self.illegal_action_penalty = 1 if \
            self.devices_settings['heating_lvl'] == 0 else 0

        self.devices_settings['heating_lvl'] = round(
            truncate(self.devices_settings['heating_lvl'] - self.influence), 2)

    def action_more_light(self):
        """Action to be taken by RL-agent"""
        self.illegal_action_penalty = 1 if \
            self.devices_settings['light_lvl'] == 1 else 0

        self.devices_settings['light_lvl'] = round(
            truncate(self.devices_settings['light_lvl'] + self.influence), 2)

    def action_less_light(self):
        """Action to be taken by RL-agent"""
        self.illegal_action_penalty = 1 if \
            self.devices_settings['light_lvl'] == 0 else 0

        self.devices_settings['light_lvl'] = round(
            truncate(self.devices_settings['light_lvl'] - self.influence), 2)

    def action_curtains_down(self):
        """Action to be taken by RL-agent"""
        self.illegal_action_penalty = 1 if \
            self.devices_settings['curtains_lvl'] == 1 else 0

        self.devices_settings['curtains_lvl'] = round(
            truncate(self.devices_settings['curtains_lvl'] + self.influence), 2)

    def action_curtains_up(self):
        """Action to be taken by RL-agent"""
        self.illegal_action_penalty = 1 if \
            self.devices_settings['curtains_lvl'] == 0 else 0

        self.devices_settings['curtains_lvl'] = round(
            truncate(self.devices_settings['curtains_lvl'] - self.influence), 2)

    def action_nop(self):
        """Action to be taken by RL-agent - do nothing"""
        self.illegal_action_penalty = 0
