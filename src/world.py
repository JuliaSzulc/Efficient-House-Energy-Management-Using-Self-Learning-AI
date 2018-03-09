from datetime import datetime, timedelta


class World:
    """Time and weather computations"""
    
    def __init__(self):
        # time settings
        self.start_date = datetime(2020, 1, 1, 0, 0, 0)   
        self.current_date = self.start_date
        self.daytime = None
        self.time_step = timedelta(minutes=60)
        self.stop_date = self.start_date + timedelta(days = 1)
        
        self.compute_daytime()

        # TODO: weather settings
        self.weather = dict()
        # other settings
        self.listeners = []
        
    def register(self, listener):
        self.listeners.append(listener)

    def update_listeners(self):
        for listener in self.listeners:
            try:
                listener.update(self.daytime, self.weather)
            except AttributeError:
                print('listener has unimplemented method update')
                    
    def step(self):
        """Proceed one step in time, collect info and update listeners"""

        if self.current_date >= self.stop_date:
            raise ValueError('end of simulation')
        
        self.current_date += self.time_step
        self.compute_daytime()
        self.update_weather()
        self.update_listeners()
        
    def compute_daytime(self):
        now = self.current_date
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.daytime = (now - midnight).seconds // 60

    def update_weather(self):
        # TODO: run all weather methods in proper order
        # and update self.weather
        pass