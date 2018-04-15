"""This module provides the House utilities

in form of a House class and additional functions. House class is the most
important part of HouseEnergyEnvironment structure, in which most of the actual
actions take place. It simulates light and energy distribution and cost,
contains actions for RL agent to change inside parameters such as heating or
light levels; it also calculates reward / penalty for the agent.

Is is mainly used within HouseEnergyEnvironment class, and should not be used
directly from outside.

"""
from random import uniform
from collections import OrderedDict
from tools import truncate


class House:
    """Main environment part"""

    def __init__(self, timeframe):
        #  --- Time ---
        # values are expressed in minutes
        self.timeframe = timeframe
        self.day_start = 7 * 60
        self.day_end = 18 * 60
        self.daytime = 0

        #  --- Energy / Light settings ---
        self.pv_absorption = 125  # Watt on max sun intensity (growth on 1 min)
        self.grid_cost = 0.5  # PLN for 1kWh
        self.house_isolation_factor = 0.998
        self.house_light_factor = 0.01
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

        #  --- Requests ---
        # calculation of 'light_desired':
        # 200 / (max_outside_illumination * house_light_factor
        #        + max_led_illuminance)
        self.user_requests = {
            'day': OrderedDict({
                'temp_desired': 21,
                'temp_epsilon': 0.5,
                'light_desired': 0.4,
                'light_epsilon': 0.05
            }),
            'night': OrderedDict({
                'temp_desired': 18,
                'temp_epsilon': 1,
                'light_desired': 0.0,
                'light_epsilon': 0.05
            })
        }

        #  --- Sensors ---
        self.inside_sensors = OrderedDict({
            'first': OrderedDict({
                'temperature': 18,
                'temperature_delta': 0,
                'light': 0
            })
        })

        #  --- Action-controlled settings ---
        self.devices_settings = OrderedDict({
            'energy_src': 'grid',  # grid/pv
            'cooling_lvl': 0,
            'heating_lvl': 0,
            'light_lvl': 0,
            'curtains_lvl': 0
        })

        # actions influence on current settings - default to 0.2 / min
        self.influence = 0.2 * timeframe

    def _calculate_light(self, outside_illumination):
        # probably should include daytime (angle of the sunlight)
        for data in self.inside_sensors.values():
            light = ((outside_illumination * self.house_light_factor)
                     * (1 - self.devices_settings['curtains_lvl'])
                     + self.devices_settings['light_lvl']
                     * self.max_led_illuminance) / self.max_led_illuminance

            data['light'] = truncate(light)

    def _calculate_temperature(self, outside_temp):
        """Calculates new temperature inside"""
        for data in self.inside_sensors.values():
            last_inside_temp = data['temperature']
            temp_delta = (outside_temp - last_inside_temp) \
                * (1 - self.house_isolation_factor)

            new_inside_temp = last_inside_temp \
                + self.timeframe \
                * (temp_delta + self.devices_settings['heating_lvl']
                    - self.devices_settings['cooling_lvl']) / 5

            data['temperature_delta'] = new_inside_temp - last_inside_temp
            data['temperature'] = new_inside_temp

    def _calculate_accumulated_energy(self, outside_light):
        """Calculates new value of energy accumulated in the battery"""
        acc = outside_light * self.pv_absorption * self.timeframe
        self.battery['delta'] = acc
        self.battery['current'] = truncate(arg=(acc + self.battery['current']),
                                           upper=self.battery['max'])

    def update(self, sensor_out_info):
        """Updates house parameters

        Args:
            sensor_out_info (dict) - weather and time information from
                                     outside sensor
        """

        self.daytime = sensor_out_info['daytime']
        self._calculate_accumulated_energy(sensor_out_info['light'])
        self._calculate_temperature(sensor_out_info['actual_temp'])
        self._calculate_light(sensor_out_info['light']
                              * self.max_outside_illumination)

    def get_inside_params(self):
        """Returns all important information about the state of the house

        Returns:
            inside_params (dict): A dictionary with *unnormalized*
            house info and noise.

        Structure of returned dict consist of:
            'inside_sensors' - dict of inside sensors info
            'desired' - dict of settings requested by user
            (currently not included) 'grid_cost' - a cost of energy
            'battery_level' - current battery level
            'battery_delta' - accumulation tempo of the battery
        """

        inside_params = OrderedDict({
            'inside_sensors': self.inside_sensors,
            'desired': self._get_current_user_requests(),

            # FIXME change grid cost to be dependent on daytime
            # FIXME no point in returning something that is always constant
            # 'grid_cost': self.grid_cost,
            'devices_settings': self.devices_settings,
            'battery_level': self.battery['current'],
            'battery_delta': self.battery['delta']
        })

        for sensor in inside_params['inside_sensors'].values():
            for key, value in sensor.items():
                if key == 'temperature':
                    sensor[key] = truncate(value
                                           + uniform(-0.05, 0.05), -20, 40)
                elif key == 'light':
                    sensor[key] = truncate(value
                                           + uniform(-0.01, 0.01))

        return inside_params

    def _calculate_device_energy_usage(self, device_full_power,
                                       device_setting):
        """
        Calculates cost of last time-frame's energy usage of given device

        Args:
            device_full_power(numeric): full potential power in kWh
            device_setting(numeric): value from 0 to 1
        Returns:
            energy usage of last timeframe expressed in kWh
        """

        return device_full_power * device_setting * (self.timeframe / 60)

    def _calculate_energy_cost(self):

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
                self.battery['current'] = \
                    truncate(self.battery['current'] - usage,
                             0, self.battery['max'])
                return 0
            else:
                # if we have used MORE energy than we had in battery, THEN we
                # switch energy_source, but we still return the cost of extra
                # amount, not 0!
                usage -= self.battery['current']
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

        w_temp, w_light, w_cost = 1.5, 10.0, 0.2
        temp_exponent, light_exponent = 2, 2

        cost = self._calculate_energy_cost()
        temp, light = (self.inside_sensors['first']['temperature'],
                       self.inside_sensors['first']['light'])
        req = self._get_current_user_requests()

        temp_penalty = self._calculate_penalty(temp,
                                               req['temp_desired'],
                                               req['temp_epsilon'],
                                               temp_exponent)
        light_penalty = self._calculate_penalty(light,
                                                req['light_desired'],
                                                req['light_epsilon'],
                                                light_exponent)

        reward = -1 * ((cost * w_cost)
                       + (temp_penalty * w_temp)
                       + (light_penalty * w_light))

        return reward / 500

    def _get_current_user_requests(self):
        """
        Returns:
             requests(dict): user requests corresponding to current time
        """

        if self.day_start <= self.daytime < self.day_end:
            return self.user_requests['day']
        return self.user_requests['night']

    @staticmethod
    def _calculate_penalty(current, desired, epsilon, power):
        """
        Returns:
             penalty(float): penalty for difference between current
             and desired param (with epsilon-acceptable consideration)
        """

        difference = abs(current - desired)
        if difference > epsilon:
            return pow(difference, power)
        return 0

    # from this point, define house actions.
    # IMPORTANT! All action names (and only them) have to start with "action"!

    def action_source_grid(self):
        """Action to be taken by RL-agent - change power source"""
        self.devices_settings['energy_src'] = 'grid'

    def action_source_battery(self):
        """Action to be taken by RL-agent - change power source"""
        # only if battery is more than 40%
        # NOTE: 0.4 is a constant. Consider moving it to come config XML / JSON
        if self.battery['current'] >= 0.4 * self.battery['max']:
            self.devices_settings['energy_src'] = 'battery'

    def action_more_cooling(self):
        """Action to be taken by RL-agent"""
        self.devices_settings['cooling_lvl'] = \
            truncate(self.devices_settings['cooling_lvl'] + self.influence)

    def action_less_cooling(self):
        """Action to be taken by RL-agent"""
        self.devices_settings['cooling_lvl'] = \
            truncate(self.devices_settings['cooling_lvl'] - self.influence)

    def action_more_heating(self):
        """Action to be taken by RL-agent"""
        self.devices_settings['heating_lvl'] = \
            truncate(self.devices_settings['heating_lvl'] + self.influence)

    def action_less_heating(self):
        """Action to be taken by RL-agent"""
        self.devices_settings['heating_lvl'] = \
            truncate(self.devices_settings['heating_lvl'] - self.influence)

    def action_more_light(self):
        """Action to be taken by RL-agent"""
        self.devices_settings['light_lvl'] = \
            truncate(self.devices_settings['light_lvl'] + self.influence)

    def action_less_light(self):
        """Action to be taken by RL-agent"""
        self.devices_settings['light_lvl'] = \
            truncate(self.devices_settings['light_lvl'] - self.influence)

    def action_curtains_down(self):
        """Action to be taken by RL-agent"""
        self.devices_settings['curtains_lvl'] = \
            truncate(self.devices_settings['curtains_lvl'] + self.influence)

    def action_curtains_up(self):
        """Action to be taken by RL-agent"""
        self.devices_settings['curtains_lvl'] = \
            truncate(self.devices_settings['curtains_lvl'] - self.influence)

    def action_nop(self):
        """Action to be taken by RL-agent - do nothing"""
        pass
