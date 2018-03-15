class House():

    """Main environment part"""

    def __init__(self):

        self.day_start = 7 * 60
        self.day_end = 18 * 60
        self.pv_absorption = 2000  # Watt on max sun intensity. pv->photovoltaic
        self.grid_cost = 0.5  # PLN for 1kWh
        self.battery = {
            'current': 0,
            'max': 14000  # Watt, thats as good as single Tesla PowerWall unit.
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
            'first' : {
                'temperature': 0,
                'light': 0
            }
        }

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

    def _calculate_energy_cost(self):
        # TODO: implement me!
        return 0

    def reward(self):
        """
        Calculate reward for the last timeframe. Note that the reward in the whole simulator is always non-positive,
        so it is easier to interpret as penalty in this case.

        Function is parametrized by weights for every factor's penalty and by exponents (for temp and light penalties)
        To see how exponents are used, check _calculate_penalty() method

        Returns:
             weighted sum of penalties
        """

        w_temp, w_light, w_cost = 1.0, 1.0, 1.0
        temp_exponent, light_exponent = 2, 2

        cost = self._calculate_energy_cost()
        temp, light = self.inside_sensors['first']['temperature'], self.inside_sensors['first']['light']
        req = self._get_current_user_requests()

        temp_penalty = self._calculate_penalty(temp, req['temp_desired'], req['temp_epsilon'], temp_exponent)
        light_penalty = self._calculate_penalty(light, req['light_desired'], req['light_epsilon'], light_exponent)

        return -1 * ((cost * w_cost) + (temp_penalty * w_temp) + (light_penalty * w_light))

    def _get_current_user_requests(self):
        """
        Returns:
             user requests corresponding to current time (day or night)
        """

        if self.day_start <= self.daytime < self.day_end:
            return self.user_requests['day']
        else:
            return self.user_requests['night']

    @staticmethod
    def _calculate_penalty(current, desired, epsilon, power):
        """
        Returns:
             penalty for difference between current and desired param (with epsilon-acceptable consideration)
        """

        difference = current - desired
        if abs(difference) > epsilon:
            return pow(difference, power)
        else:
            return 0

    # from this point, define house actions. 
    # IMPORTANT! All action names (and only them ) have to start with "action"!

    def action_switch_energy_src(self):
        # TODO: implement me!
        pass

    def action_more_cooling(self):
        #TODO: implement me!
        pass

    def action_less_cooling(self):
        #TODO implement me!
        pass
    
    def action_more_heating(self):
        #TODO implement me!
        pass
    
    def action_less_heating(self):
        #TODO implement me!
        pass
    
    def action_more_light(self):
        #TODO implement me!
        pass
    
    def action_less_light(self):
        #TODO implement me!
        pass
    
    def action_curtains_up(self):
        #TODO implement me!
        pass
    
    def action_curtains_down(self):
        #TODO implement me!
        pass
    
    def action_nop(self):
        pass


                    
