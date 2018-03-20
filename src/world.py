from datetime import datetime, timedelta


class World:
    """Time and weather computations"""

    def __init__(self):
        # time settings
        self.start_date = datetime(2020, 1, 1, 0, 0, 0)
        self.current_date = self.start_date
        self.daytime = None
        self.time_step = timedelta(minutes=60)
        self.timeframe_minutes = self.time_step.seconds // 60
        self.stop_date = self.start_date + timedelta(days = 1)

        self._compute_daytime()

        # weather part
        self.weather = {
            'temp': 0,
            'sun': 0,
            'clouds': 0,
            'rain': 0,
            'wind': 0
        }

        # other settings
        self.listeners = []

    def register(self, listener):
        self.listeners.append(listener)

    def _update_listeners(self):
        for listener in self.listeners:
            try:
                listener.update(daytime=self.daytime, weather=self.weather)
            except AttributeError:
                print('listener has unimplemented method update')

    def step(self):
        """Proceed one step in time, collect info and update listeners
        Returns:
            done(boolean): information if the state after the step is terminal (episode end).

        """

        if self.current_date >= self.stop_date:
            return True

        self.current_date += self.time_step
        self._compute_daytime()
        self._update_weather()
        self._update_listeners()
        return False

    def _compute_daytime(self):
        now = self.current_date
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.daytime = (now - midnight).seconds // 60


    def _update_weather(self):
        # TODO: run all weather methods in proper order
        # and update self.weather
        pass

    # from this point, only weather methods

    def _calculate_sun(self):
        # TODO: implement me!
        pass

    def _calculate_wind(self):
        # TODO: implement me!
        pass

    def _calculate_clouds(self):
        # TODO: implement me!
        pass

    def _calculate_rain(self):
        # TODO: implement me!
        pass

    def _calculate_temperature(self):
        # TODO: implement me!
        pass

