class House():

    """Main environment part"""

    def __init__(self):

        self.pv_absorption = 2000 # Watt on max sun intensity. pv->photovoltaic
        self.grid_cost = 0.5 # PLN for 1kWh
        self.battery = {
            'current' : 0,
            'max' : 14000 # Watt, thats as good as single Tesla PowerWall unit.
        }

        self.user_requests = {
            'day' : {
                'temp_min' : 20,
                'temp_max' : 22,
                'light_min' : 0.7 
            },
            'night' : {
                'temp_min' : 17,
                'temp_max' : 20,
            }
        }

        self.inside_sensors = {
            'first' : {
                'temperature' : 0,
                'light' : 0
            }
        }

        self.current_settings = {
            'energy_src' : 'grid',
            'cooling_lvl' : 0,
            'heating_lvl' : 0,
            'light_lvl' : 0,
            'curtains_lvl' : 0 
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
        pass

    def reward(self):
        #TODO: Given values from inside sensors,energy costs and user requests,
        # calculate reward
        reward = 0
        return reward
        

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
                    
