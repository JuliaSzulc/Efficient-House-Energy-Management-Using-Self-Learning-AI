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
from random import randint
from collections import OrderedDict


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


class House:
    """Main environment part"""

    def __init__(self, timeframe):
        # --- TIME house settings in minutes ---
        self.timeframe = timeframe
        self.day_start = 7 * 60
        self.day_end = 18 * 60
        self.daytime = None  # current time

        # --- ENERGY / LIGHT house settings ---
        self.pv_absorption = 125  # Watt on max sun intensity (growth on 1 min)
        self.grid_cost = 0.5  # PLN for 1kWh
        self.house_isolation_factor = 0.9
        self.house_light_factor = 0.01
        self.max_led_illuminance = 200  # lux
        self.battery = {
            'current': 0,
            'max': 14000  # Watt, as good as single Tesla PowerWall unit.
        }
        self.devices_power = {
            'air_conditioner': 1500,
            'heater': 3000,
            'light': 20
        }

        #  --- REQUESTS - user settings ---
        # calculation of 'light_desired':
        # 200 / (25000 * self.house_light_factor + self.max_led_illuminance)
        self.user_requests = OrderedDict({
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
        })

        # --- SENSORS indications ---
        self.inside_sensors = OrderedDict({
            'first': OrderedDict({
                'temperature': randint(15, 25),
                'light': 0
            })
        })

        # --- ACTIONS-controlled settings, to be used by RL-agent ---
        self.current_settings = {
            'energy_src': 'grid',  # grid/pv
            'cooling_lvl': 0,
            'heating_lvl': 0,
            'light_lvl': 0,
            'curtains_lvl': 0
        }
        # actions influence on current settings - default to 0.2 / min
        self.influence = 0.2 * timeframe

    def _calculate_light(self, outside_illumination):
        # probably should include daytime (angle of the sunlight)
        for data in self.inside_sensors.values():
            light = ((outside_illumination * self.house_light_factor)
                     * (1 - self.current_settings['curtains_lvl'])
                     + self.current_settings['light_lvl']
                     * self.max_led_illuminance) / self.max_led_illuminance

            data['light'] = truncate(light)

    def _calculate_temperature(self, actual_temp):
        # as long as we implement only one heater, the inside temperature
        # is average temperature from all inside sensors
        temperatures = [d['temperature'] for d in self.inside_sensors.values()]
        inside_temp = sum(temperatures) / len(temperatures)

        temp_delta = (actual_temp - inside_temp) \
                     * (1 - self.house_isolation_factor)

        # should be changed for some more complex formula
        for data in self.inside_sensors.values():
            temperature = inside_temp + (temp_delta * self.timeframe / 50) + \
                          + (self.timeframe
                             * self.current_settings['heating_lvl'] / 10) \
                          - (self.timeframe
                             * self.current_settings['cooling_lvl'] / 10)

            # print("Inside: ", temperature, "  | Outside: ", actual_temp)
            data['temperature'] = temperature

    def _calculate_accumulated_energy(self, outside_light):
         # outside_light is normalized light from world [0, 1]
        # acc is value describes battery power growth in one full step where
        # it can rise pv_absorption per minute maximum

        acc = outside_light * self.pv_absorption \
              * self.timeframe

        self.battery['current'] = truncate(
            arg=(acc + self.battery['current']),
            upper=self.battery['max'])

    def update(self, sensor_out_info):
        """Updates house parameters

        Args:
            sensor_out_info (dict) - weather and time informations from
                                     outside sensor

        """

        self.daytime = sensor_out_info['daytime']
        self._calculate_accumulated_energy(sensor_out_info['light'])
        self._calculate_temperature(sensor_out_info['actual_temp'])
        self._calculate_light(sensor_out_info['illumination'])

    def get_inside_params(self):
        """All important house informations, together

        Returns:
            inside_params (dict): A dictionary with house info, with additional
                                  noise.

        Structure of returned dict consist of:
            'inside_sensors' - dict of inside sensors info
            'desired' - dict of settings requested by user
            'grid_cost' - a cost of energy
            'battery_level' - current battery level

        """

        # NOTE: when making changes, make sure every nested dict is Ordered!
        inside_params = OrderedDict({
            'inside_sensors': self.inside_sensors,
            'desired': self.user_requests,
            'grid_cost': self.grid_cost,
            'battery_level': self.battery['current']
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

    def _calculate_device_cost(self, device_power, device_settings):
        # 1000 means kilo, like in kiloWatt hours, and 60 is for mins in hour
        return device_power / 1000 / 60 * device_settings * self.timeframe \
               * self.grid_cost

    def _calculate_energy_cost(self):
        if self.current_settings['energy_src'] == 'pv':
            return 0

        cost = self._calculate_device_cost(
            self.devices_power['air_conditioner'],
            self.current_settings['cooling_lvl']
        )
        cost += self._calculate_device_cost(
            self.devices_power['heater'],
            self.current_settings['heating_lvl']
        )
        cost += self._calculate_device_cost(
            self.devices_power['light'],
            self.current_settings['light_lvl']
        )

        return cost

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

        w_temp, w_light, w_cost = 1.0, 5.0, 50.0
        temp_exponent, light_exponent = 1.1, 2

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

        return reward / 30

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
    # IMPORTANT! All action names (and only them ) have to start with "action"!

    def action_source_grid(self):
        """Action to be taken by RL-agent - change power source"""
        self.current_settings['energy_src'] = 'grid'

    def action_source_battery(self):
        """Action to be taken by RL-agent - change power source"""
        # only if battery is more than 40%
        if self.battery['current'] >= 0.4 * self.battery['max']:
            self.current_settings['energy_src'] = 'battery'

    def action_more_cooling(self):
        """Action to be taken by RL-agent"""
        self.current_settings['cooling_lvl'] = \
            truncate(self.current_settings['cooling_lvl'] + self.influence)

    def action_less_cooling(self):
        """Action to be taken by RL-agent"""
        self.current_settings['cooling_lvl'] = \
            truncate(self.current_settings['cooling_lvl'] - self.influence)

    def action_more_heating(self):
        """Action to be taken by RL-agent"""
        self.current_settings['heating_lvl'] = \
            truncate(self.current_settings['heating_lvl'] + self.influence)

    def action_less_heating(self):
        """Action to be taken by RL-agent"""
        self.current_settings['heating_lvl'] = \
            truncate(self.current_settings['heating_lvl'] - self.influence)

    def action_more_light(self):
        """Action to be taken by RL-agent"""
        self.current_settings['light_lvl'] = \
            truncate(self.current_settings['light_lvl'] + self.influence)

    def action_less_light(self):
        """Action to be taken by RL-agent"""
        self.current_settings['light_lvl'] = \
            truncate(self.current_settings['light_lvl'] - self.influence)

    def action_curtains_down(self):
        """Action to be taken by RL-agent"""
        self.current_settings['curtains_lvl'] = \
            truncate(self.current_settings['curtains_lvl'] + self.influence)

    def action_curtains_up(self):
        """Action to be taken by RL-agent"""
        self.current_settings['curtains_lvl'] = \
            truncate(self.current_settings['curtains_lvl'] - self.influence)

    def action_nop(self):
        """Action to be taken by RL-agent - do nothing"""
        pass
