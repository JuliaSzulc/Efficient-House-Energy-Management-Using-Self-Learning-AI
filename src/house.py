"""This module provides the House utilities

in form of a House class and additional functions. House class is the most
important part of HouseEnergyEnvironment structure, in which most of the actual
actions take place. It simulates light and energy distribution and their cost;
contains actions for RL agent to change inside parameters such as heating or
light levels; it also calculates the reward / penalty for the last timeframe.

Is is mainly used within HouseEnergyEnvironment class, and should not be used
directly from outside.

Note: Currently, the sensor has no specific impact on the registered values, so
every sensor would register the same values. There is only one, main sensor
used at the moment.

"""
import json
import os
import random
from collections import OrderedDict
from tools import truncate


class House:
    """Main environment part"""

    def __init__(self, timeframe):
        """Initialize the House object

        Args:
            timeframe(numeric): duration of timeframe in minutes.

        House object fields include:

        - time constants/variables, expressed in minutes, to define the
          day/night time, current time
        - Energy, light, isolation constants. This includes empirically chosen
          isolation and light factors, absorption of photovoltaic battery.
          The max illuminance fields are real-world based, shouldn't be changed
        - Photovoltaic Battery parameters - current accumulation, delta and max
          capacity.
        - Devices' power. Currently the light power is unrealistically
          increased for better agent performance and this should be fixed
          in other way.
        - User Requests about the desired temperature and light. Note that
          epsilons (they define 'acceptance intervals') are currently
          not used in the reward function, as they seem to make the
          training harder.
        - Inside Sensors, which register parameters. There is only one, main
          sensor currently.
        - Devices action-controlled settings of <0.0, 1.0> values
        - Action penalty field used to express the current penalty for using
          an 'illegal' actions. You can use the field to penalize unwanted
          behaviours, like using some actions as NOP action etc.
        - Influence, describing how large is the change of device setting
          when action is executed. Note that the influence for light-related
          actions is influence divided by 2 to allow better precision.
        """

        add_path = ''
        if 'tests' in os.getcwd():
            add_path = '../'
        with open(add_path + '../configuration.json') as config_file:
            self.config = json.load(config_file)['env']

        self.timeframe = timeframe
        self.day_start = self.config['day_start']
        self.day_end = self.config['day_end']
        self.daytime = 0

        self.max_pv_absorption = self.config['max_pv_absorption']
        self.grid_cost = self.config['night_grid_cost']
        self.house_isolation_factor = self.config['house_isolation_factor']
        self.house_light_factor = self.config['house_light_factor']
        self.max_led_illumination = 200  # lux
        self.max_outside_illumination = 25000  # lux
        self.battery = {
            'current': 0,
            'delta': 0,
            'max': self.config['battery_max']  # Watt
        }
        self.devices_power = self.config['devices_power']

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

        self.devices_settings = OrderedDict({
            'energy_src': 'grid',  # grid/battery
            'cooling_lvl': 0,
            'heating_lvl': 0,
            'light_lvl': 0,
            'curtains_lvl': 0
        })

        self.action_penalty = 0

        self.influence = self.config['influence_per_min'] * timeframe

    def _update_grid_cost(self):
        """Updates the grid cost based on daytime."""
        if self.day_start < self.daytime < self.day_end:
            self.grid_cost = self.config['day_grid_cost']
        else:
            self.grid_cost = self.config['night_grid_cost']

    def _update_user_requests(self):
        """Randomly updates the user requests every 2 hours

        For the temperature, requests are integers from <19, 24> interval
        during the day, and from <15, 20> interval during the night time.

        For the light, interval remain the same during the whole time - <0, 1>
        Values are rounded to 1 decimal point.
        """

        if self.daytime % 120 == 0:
            self.user_requests['temp_desired'] = random.randint(15, 20)

            if self.day_start < self.daytime < self.day_end:
                self.user_requests['temp_desired'] += 4

            self.user_requests['light_desired'] = round(random.random(), 1)

    def _calculate_light(self, outside_illumination):
        """Updates the inside sensors with new light value.

        Final light is normalized to <0, 1> interval and depends on the
        light device level, curtains level, outside light level and
        house_light_factor, which describes how much illumination 'enters'
        the house.

        Args:
            outside_illumination(numeric): Registered value of outside
                                           illumination

        Note: Currently, the sensor has no impact on the registered value, so
        every sensor would register the same value.
        """

        for data in self.inside_sensors.values():
            outside_light = (outside_illumination * self.house_light_factor) \
                * (1 - self.devices_settings['curtains_lvl'])

            inside_light = self.devices_settings['light_lvl'] \
                * self.max_led_illumination

            final_light = (outside_light + inside_light) \
                / self.max_led_illumination

            data['light'] = truncate(final_light)

    def _calculate_temperature(self, outside_temp):
        """Updates the inside sensors with new temperature value.

        The new value depends on the last temperature, the outside temperature,
        house's isolation factor, and levels of heating and cooling devices.
        Value is truncated to <-20, 40> interval and this interval is assumed
        in the environment's normalization method -
        HouseEnergyEnvironment.serialize_state().

        Args:
            outside_temp(numeric): Unnormalized value of registered
                                   outside temperature

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

    def _calculate_accumulated_energy(self, outside_illumination):
        """Calculates new value of energy accumulated in the battery

        Args:
            outside_illumination(numeric): Registered value of outside
                                           illumination
        """
        acc = outside_illumination * self.max_pv_absorption * self.timeframe
        self.battery['delta'] = acc
        self.battery['current'] = truncate(arg=(acc + self.battery['current']),
                                           upper=self.battery['max'])

    def update(self, sensor_out_info):
        """Updates house parameters. The order of the methods matters and it
        shouldn't be changed without any good reason.

        Args:
            sensor_out_info(dict): weather and time information from
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
            'battery_delta' - accumulation speed of the battery
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
        """Calculates cost of last time-frame's energy usage of given device

        Args:
            device_full_power(numeric): full device power in kWh
            device_setting(numeric): value from 0 to 1
        Returns:
            energy usage of last timeframe expressed in kWh
        """

        return device_full_power * device_setting * (self.timeframe / 60)

    def _calculate_cost_and_update_energy_source(self):
        """Calculates the energy cost of last time-frame and updates source.

        Updating the source means subtracting the used energy from battery
        and/or switching the source. Energy used from photovoltaic battery
        is free. If the usage exceeded the battery level, the source of energy
        is switched to grid.
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
                usage -= self.battery['current']
                self.battery['current'] = 0
                self.devices_settings['energy_src'] = 'grid'

        return usage * self.grid_cost

    def reward(self):
        """Calculate reward for the last timeframe.

        Note that the reward in the whole simulator is always non-positive,
        so it is easier to interpret as penalty in this case.

        Function is parametrized by weights for every factor's penalty
        and by exponents (for temp and light penalties)
        To see how exponents are used, check _calculate_penalty() method

        Returns:
             reward(float): weighted sum of penalties plus additional action
                penalty. The final value of reward function for lats timeframe

        """

        w_temp = self.config['temperature_w_in_reward']
        w_light = self.config['light_w_in_reward']
        w_cost = self.config['cost_w_in_reward']

        cost = self._calculate_cost_and_update_energy_source()
        temp, light = (self.inside_sensors['first']['temperature'],
                       self.inside_sensors['first']['light'])
        req = self.user_requests

        temp_penalty = abs(temp - req['temp_desired'])

        light_penalty = abs(light - req['light_desired'])

        reward = (cost * w_cost) \
            + (temp_penalty * w_temp) \
            + (light_penalty * w_light)

        return -(reward + self.action_penalty)

    # All action-method names (and only them) have to start with "action"!
    # You should define the action penalty.

    def action_source_grid(self):
        """Action to be taken by RL-agent - change power source"""
        self.action_penalty = 1 if \
            self.devices_settings['energy_src'] == 'grid' else 0

        self.devices_settings['energy_src'] = 'grid'

    def action_source_battery(self):
        """Action to be taken by RL-agent - change power source"""
        self.action_penalty = 1 if \
            self.devices_settings['energy_src'] == 'battery' else 0

        self.devices_settings['energy_src'] = 'battery'

    def action_more_cooling(self):
        """Action to be taken by RL-agent. Increase cooling level."""
        self.action_penalty = 1 if \
            self.devices_settings['cooling_lvl'] == 1 else 0

        self.devices_settings['cooling_lvl'] = round(
            truncate(self.devices_settings['cooling_lvl'] + self.influence), 4)

    def action_less_cooling(self):
        """Action to be taken by RL-agent. Decrease cooling level."""
        self.action_penalty = 1 if \
            self.devices_settings['cooling_lvl'] == 0 else 0

        self.devices_settings['cooling_lvl'] = round(
            truncate(self.devices_settings['cooling_lvl'] - self.influence), 4)

    def action_more_heating(self):
        """Action to be taken by RL-agent. Increase heating level."""
        self.action_penalty = 1 if \
            self.devices_settings['heating_lvl'] == 1 else 0

        self.devices_settings['heating_lvl'] = round(
            truncate(self.devices_settings['heating_lvl'] + self.influence), 4)

    def action_less_heating(self):
        """Action to be taken by RL-agent. Decrease heating level."""
        self.action_penalty = 1 if \
            self.devices_settings['heating_lvl'] == 0 else 0

        self.devices_settings['heating_lvl'] = round(
            truncate(self.devices_settings['heating_lvl'] - self.influence), 4)

    def action_more_light(self):
        """
        Action to be taken by RL-agent. Increase lights level.
        Note that this action uses 2 times smaller influence as agent
        needs a bit more precision with the light settings.
        """

        self.action_penalty = 1 if \
            self.devices_settings['light_lvl'] == 1 else 0

        self.devices_settings['light_lvl'] = round(
            truncate(self.devices_settings['light_lvl']
                     + self.influence / 2), 4)

    def action_less_light(self):
        """
        Action to be taken by RL-agent. Decrease lights level.
        Note that this action uses 2 times smaller influence as agent
        needs a bit more precision with the light settings.
        """

        self.action_penalty = 1 if \
            self.devices_settings['light_lvl'] == 0 else 0

        self.devices_settings['light_lvl'] = round(
            truncate(self.devices_settings['light_lvl']
                     - self.influence / 2), 4)

    def action_curtains_down(self):
        """
        Action to be taken by RL-agent. Lower the curtains.
        Note that this action uses 2 times smaller influence as agent
        needs a bit more precision with the light settings.
        There is a small penalty for using this action, to prevent it
        from being used as an action_nop.
        """

        self.action_penalty = 1 if \
            self.devices_settings['curtains_lvl'] == 1 else 0.05

        self.devices_settings['curtains_lvl'] = round(
            truncate(self.devices_settings['curtains_lvl']
                     + self.influence / 2), 4)

    def action_curtains_up(self):
        """
        Action to be taken by RL-agent. Increase the curtains level.
        Note that this action uses 2 times smaller influence as agent
        needs a bit more precision with the light settings.
        There is a small penalty for using this action, to prevent it
        from being used as an action_nop.
        """

        self.action_penalty = 1 if \
            self.devices_settings['curtains_lvl'] == 0 else 0.05

        self.devices_settings['curtains_lvl'] = round(
            truncate(self.devices_settings['curtains_lvl']
                     - self.influence / 2), 4)

    def action_nop(self):
        """Action to be taken by RL-agent - do nothing"""
        self.action_penalty = 0
