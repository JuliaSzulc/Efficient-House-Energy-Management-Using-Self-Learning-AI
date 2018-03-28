from random import uniform

class House:

    """Main environment part"""

    def __init__(self, timeframe):
        self.timeframe = timeframe # in minutes!

        self.day_start = 7 * 60
        self.day_end = 18 * 60
        self.pv_absorption = 2000  # Watt on max sun intensity
        self.grid_cost = 0.5  # PLN for 1kWh
        self.house_isolation_factor = 0.5
        self.house_light_factor = 0.01
        self.max_led_illuminance = 200  # lux
        self.battery = {
            'current': 0,
            'max': 14000  # Watt, as good as single Tesla PowerWall unit.
        }

        self.user_requests = {
            'day': {
                'temp_desired': 21,
                'temp_epsilon': 0.5,
                'light_desired': 0.4, # 200 / (25000 * self.house_light_factor + self.max_led_illuminance)
                'light_epsilon': 0.1
            },
            'night': {
                'temp_desired': 18,
                'temp_epsilon': 1,
                'light_desired': 0.0,
                'light_epsilon': 0.1
            }
        }

        self.inside_sensors = {
            'first': {
                'temperature': 0,
                'light': 0
            }
        }

        # o tę ilość będziemy zmieniać parametry za pomocą akcji
        # poczatkowo zakładamy 0.2 na minutę
        self.influence = 0.2 * timeframe

        self.current_settings = {
            'energy_src': 'grid', # grid/pv
            'cooling_lvl': 0,
            'heating_lvl': 0,
            'light_lvl': 0,
            'curtains_lvl': 0
        }

        self.devices_power = {
            'air_conditioner': 1500,
            'heater': 3000,
            'light': 20
        }

        self.daytime = None

    def calculate_light(self, outside_illumination):
        # probably should include daytime (angle of the sunlight)
        for sensor, data in self.inside_sensors.items():
            data['light'] = ((outside_illumination * self.house_light_factor)
                * (1 - self.current_settings['curtains_lvl'])
                + self.current_settings['light_lvl'] *
                self.max_led_illuminance) / self.max_led_illuminance

    def calculate_temperature(self, actual_temp):
        # as long as we implement only one heater the inside temperature is average
        # temperature from all inside sensors
        inside_temp = 0

        for sensor, data in self.inside_sensors.items():
            inside_temp += data['temperature']

        inside_temp /= len(self.inside_sensors.items())

        temp_delta = abs((actual_temp - inside_temp)
            * (1 - self.house_isolation_factor))

        # should be changed for some more complex formula
        for data in self.inside_sensors.values():
            data['temperature'] = inside_temp + (temp_delta * self.timeframe
                * self.current_settings['heating_lvl']) - (temp_delta
                * self.timeframe * self.current_settings['cooling_lvl'])

    def calculate_accumulated_energy(self, outside_illumination):
        accumulated_energy = outside_illumination * self.pv_absorption \
            * self.timeframe

        if self.battery['current'] + accumulated_energy <= self.battery['max']:
            self.battery['current'] += accumulated_energy;

    def update(self, sensor_out_info):
        self.daytime = sensor_out_info['daytime']
        self.calculate_accumulated_energy(sensor_out_info['light'])
        self.calculate_temperature(sensor_out_info['actual_temp'])
        self.calculate_light(sensor_out_info['illumination'])

    def get_inside_params(self):
        inside_params = {
            'inside_sensors': self.inside_sensors,
            'desired': self.user_requests,
            'grid_cost': self.grid_cost,
            'battery_level': self.battery['current']
        }

        for sensor in inside_params['inside_sensors'].values():
            for parameter in sensor.keys():
                sensor[parameter] += uniform(-0.1, 0.1)

        return inside_params

    def calculate_device_cost(self, device_power, device_settings):
        return device_power / 1000 / 60 * device_settings * self.timeframe \
            * self.grid_cost

    def _calculate_energy_cost(self):
        if self.current_settings['energy_src'] == 'pv':
            return 0

        cost = self.calculate_device_cost(
            self.devices_power['air_conditioner'],
            self.current_settings['cooling_lvl'])
        cost += self.calculate_device_cost(self.devices_power['heater'],
            self.current_settings['heating_lvl'])
        cost += self.calculate_device_cost(self.devices_power['light'],
            self.current_settings['light_lvl'])

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

        w_temp, w_light, w_cost = 1.0, 1.0, 1.0
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

        return reward

    def _get_current_user_requests(self):
        """
        Returns:
             requests(dict): user requests corresponding to current time
        """

        if self.day_start <= self.daytime < self.day_end:
            return self.user_requests['day']
        else:
            return self.user_requests['night']

    @staticmethod
    def _calculate_penalty(current, desired, epsilon, power):
        """
        Returns:
             penalty(float): penalty for difference between current
             and desired param (with epsilon-acceptable consideration)
        """

        difference = current - desired
        if abs(difference) > epsilon:
            return pow(difference, power)
        else:
            return 0

    # from this point, define house actions.
    # IMPORTANT! All action names (and only them ) have to start with "action"!

    def action_source_grid(self):
        self.current_settings['energy_src'] = 'grid'

    def action_source_battery(self):
        self.current_settings['energy_src'] = 'battery'

    def action_more_cooling(self):
        self.current_settings['cooling_lvl'] += self.influence
        if self.current_settings['cooling_lvl'] > 1:
            self.current_settings['cooling_lvl'] = 1

    def action_less_cooling(self):
        self.current_settings['cooling_lvl'] -= self.influence
        if self.current_settings['cooling_lvl'] < 0:
            self.current_settings['cooling_lvl'] = 0

    def action_more_heating(self):
        self.current_settings['heating_lvl'] += self.influence
        if self.current_settings['heating_lvl'] > 1:
            self.current_settings['heating_lvl'] = 1

    def action_less_heating(self):
        self.current_settings['heating_lvl'] -= self.influence
        if self.current_settings['heating_lvl'] < 0:
            self.current_settings['heating_lvl'] = 0

    def action_more_light(self):
        self.current_settings['light_lvl'] += self.influence
        if self.current_settings['light_lvl'] > 1:
            self.current_settings['light_lvl'] = 1

    def action_less_light(self):
        self.current_settings['light_lvl'] -= self.influence
        if self.current_settings['light_lvl'] < 0:
            self.current_settings['light_lvl'] = 0

    def action_curtains_down(self):
        self.current_settings['curtains_lvl'] += self.influence
        if self.current_settings['curtains_lvl'] > 1:
            self.current_settings['curtains_lvl'] = 1

    def action_curtains_up(self):
        self.current_settings['curtains_lvl'] -= self.influence
        if self.current_settings['curtains_lvl'] < 0:
            self.current_settings['curtains_lvl'] = 0

    def action_nop(self):
        pass
