class House:

    """Main environment part"""

    def __init__(self, timeframe):
        self.timeframe = timeframe # in minutes!

        self.day_start = 7 * 60
        self.day_end = 18 * 60
        self.pv_absorption = 2000  # Watt on max sun intensity
        self.grid_cost = 0.5  # PLN for 1kWh
        self.battery = {
            'current': 0,
            'max': 14000  # Watt, as good as single Tesla PowerWall unit.
        }

        self.user_requests = {
            'day': {
                'temp_desired': 21,
                'temp_epsilon': 0.5,
                'light_desired': 0.7,
                'light_epsilon': 0.05
            },
            'night': {
                'temp_desired': 18,
                'temp_epsilon': 1,
                'light_desired': 0.0,
                'light_epsilon': 0.01
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
            'energy_src': 'grid',
            'cooling_lvl': 0,
            'heating_lvl': 0,
            'light_lvl': 0,
            'curtains_lvl': 0
        }

        self.daytime = None
        self.weather = None

    def update(self, weather, daytime, **kwargs):
        self.daytime = daytime
        self.weather = weather
        # TODO: this is the most important environment method.
        # I assume that actions are already DONE at this point.
        # we must carefully execute everything together now:
        # 1. calculate energy (temperature, light etc) flow from outside
        # 2. store accumulated energy (if any) in battery
        # 3. most importantly, calculate final values for inside sensor(s)

    def get_inside_params(self):
        """
        This method should be called AFTER updating the house.

        """

        # TODO: implement me!
        # Should return all inside sensors params with some random noise error
        # Note: this *has to* include: inside sensors values,
        # current desired levels of params, current grid_cost,
        # current battery level.
        # Should return it as one dictionary with parameters named nicely.
        pass

    def _calculate_energy_cost(self):
        # TODO: implement me!
        return 0

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

