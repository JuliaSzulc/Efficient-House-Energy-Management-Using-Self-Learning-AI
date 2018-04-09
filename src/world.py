"""This module provides outside world simulations.

Grouped in World class, these methods play a major role in
HouseEnergyEnvironment. World takes care of weather and time computations, and
also es them forward to all listeners - in default case, all outside
sensors.

It should be used inside environment class only.

"""
from datetime import datetime, timedelta
import math
import random
from random import choices


class World:
    """Time and weather computations"""

    def __init__(self, time_step_in_minutes=0.5, duration_days=1):
        # --- time settings ---
        self.start_date = datetime(2020, 1, 1, 0, 0, 0)
        self.current_date = self.start_date
        self.daytime = None
        self.time_step_in_minutes = time_step_in_minutes
        self.time_step = timedelta(minutes=time_step_in_minutes)
        self.stop_date = self.start_date + timedelta(days=duration_days)

        self._compute_daytime()

        # --- weather part ----
        # weather stats
        # temp  -> perceptible feeling temp
        # sun   -> sun power before calculating with clouds
        # light -> sun power after calculation
        self.weather = {
            'temp': 12,
            'sun': 0.0,
            'light': 0,
            'clouds': 0,
            'rain': 0,
            'wind': 0
        }

        self.delta_weather = {
            'temp_delta': 0,
            'sun_delta': 0,
            'light_delta': 0,
        }

        # sun power in (0,1) range. 0 between [7 PM, 5 AM],
        # 840 is shining time in minutes
        self.last_step_sun = 0

        self.sun_steps_count = int(840 / self.time_step_in_minutes)
        self.sun_steps = [
            math.sin(2 * math.pi * 0.5 * (i / self.sun_steps_count))
            for i in range(self.sun_steps_count)
        ]
        self.sun_steps.append(0.0)
        self.sun_steps_count += 1

        # max weight for weather
        self.previous_weather_weight = None
        self.current_weather_weight = None

        #  max time frame in minutes to calibrate previous & current
        # weather weights parameters ([max_time_frame_in_minutes],
        # [max_current_weather_weight])

        # FIXME: [from Michal] max time frame is a wrong concept. It should be
        # working for ANY timeframe, not asking for max possible value! For now
        # i'm setting it to maxiumum value from test cases (30min), but it's
        # only temporary - should be fixed asap
        max_frame = 30  # minutes.
        self._calculate_weathers_weights(max_frame, 0.95)

        # probability (summary 1.0) of difference wind in (0,1)
        # power range [0.0, 0.1, 0.2, ..., 1.0]
        self.wind_power = [i / 10 for i in range(11)]
        self.wind_probability = [0.4, 0.1, 0.07, 0.1, 0.05, 0.03, 0.1, 0.04,
                                 0.02, 0.05, 0.04]

        # probability (summary 1.0) of difference clouds in (0, 0.6)
        # power range [0.0, 0.1, 0.2, ..., 0.6]
        # 0.5 clouds means that the sun is half hidden
        self.clouds = [i / 10 for i in range(7)]
        self.clouds_probability = [0.2, 0.3, 0.2, 0.1, 0.07, 0.1, 0.03]

        # --- end of the weather part ---

        # other settings
        self.listeners = []

    def register(self, listener):
        """Registers listener on listeners list.

        Args:
            listener - object to be added to self.listeners. Should implement
                       update() method.

        """

        self.listeners.append(listener)

    def update_listeners(self):
        """Update all listeners with current weather and time"""
        for listener in self.listeners:
            try:
                listener.update(daytime=self.daytime, weather=self.weather)
            except AttributeError:
                raise AttributeError('listener has unimplemented method update')

    def step(self):
        """Proceed one step in time, collect info and update listeners
        Returns:
            done(boolean): information if the state after the step is terminal
                           (episode end).

        """

        if self.current_date >= self.stop_date:
            return True

        self.current_date += self.time_step
        self._compute_daytime()
        self._update_weather()
        self.update_listeners()
        return False

    def _compute_daytime(self):
        now = self.current_date
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.daytime = (now - midnight).seconds // 60

    def _calculate_weathers_weights(self, max_frame, max_weight):
        correct_frame = max_frame and max_frame >= self.time_step_in_minutes
        correct_max_weight = 0 <= max_weight <= 1

        assert correct_frame, 'incorrect max_frame value'
        assert correct_max_weight, 'incorrect max_weight value'

        self.current_weather_weight = max_weight / max_frame\
                                      * self.time_step_in_minutes
        self.previous_weather_weight = 1 - self.current_weather_weight

    def _update_weather(self):
        # 1) get sun state on the sky
        # 2) check wind power
        # 3) create clouds & calculate with wind
        # 4) start raining when clouds come
        # 5) calculate final temperature

        self._calculate_sun()
        self._calculate_wind()
        self._calculate_clouds()
        self._calculate_light()
        self._calculate_rain()
        self._calculate_temperature()

    # from this point, only weather methods

    def _calculate_sun(self):
        temp_sun = self.weather['sun']

        # sun shine between (5 AM , 7 PM) (14h)
        if 300 <= self.daytime <= 1140:
            self.weather['sun'] = self.sun_steps[self.last_step_sun]
            self.last_step_sun += 1
            self.last_step_sun %= self.sun_steps_count
        else:
            self.weather['sun'] = 0
            self.last_step_sun = 0

        self.delta_weather['sun_delta'] = self.weather['sun'] - temp_sun

    def _calculate_wind(self):
        # get random wind
        self.weather['wind'] = self.current_weather_weight\
                               * choices(self.wind_power,
                                         self.wind_probability,
                                         k=1)[0]\
                               + self.previous_weather_weight\
                               * self.weather['wind']

    def _calculate_clouds(self):
        # get random clouds
        # update clouds with wind power (stronger wind = less clouds)
        self.weather['clouds'] = self.current_weather_weight\
                                 * choices(self.clouds,
                                           self.clouds_probability,
                                           k=1)[0]\
                                 * (1 - self.weather['wind'])\
                                 + self.previous_weather_weight\
                                 * self.weather['clouds']

    def _calculate_light(self):
        temp_light = self.weather['light']

        # after clouds calculation we can count light parameter
        #FIXME - chmury nie zasłaniają słońca w 100% jak sa na 1
        self.weather['light'] = self.weather['sun']\
                                * (1 - self.weather['clouds'])

        self.delta_weather['light_delta'] = self.weather['light'] - temp_light

    def _calculate_rain(self):
        # if clouds are big enough then start raining
        if self.weather['clouds'] >= 0.4:
            self.weather['rain'] = 1
        else:
            self.weather['rain'] = 0

    def _calculate_temperature(self):
        temp_temperature = self.weather['temp']

        # calculate new temperature
        # lets say that ~30 degrees is max temperature when sun power
        # is 1.0 & also there is no clouds & wind which can change temperature
        # by 5 degrees (we don't use rain here for now) -> then:
        new_temperature = random.uniform(11.5, 12.5)\
                          + (18 * self.weather['light']
                             - 5 * self.weather['wind'])

        # then update new temperature including our weather weights
        self.weather['temp'] = self.previous_weather_weight\
                               * self.weather['temp']\
                               + self.current_weather_weight\
                               * new_temperature

        self.delta_weather['temp_delta'] = self.weather['temp']\
                                           - temp_temperature


def plot_weather():
    """Plot normalized weather graph in a single episode"""
    temp, sun, light, clouds, rain, wind = [], [], [], [], [], []

    world = World(duration_days=5)
    while not world.step():
        temp.append((world.weather['temp'] + 20) / 60)
        sun.append(world.weather['sun'])
        light.append(world.weather['light'])
        clouds.append(world.weather['clouds'])
        rain.append(world.weather['rain'])
        wind.append(world.weather['wind'])

    plt.plot(temp, ',', label='temperature')
    plt.plot(sun, label='sun')
    plt.plot(light, ',', label='light')
    plt.plot(clouds, ',', label='clouds')
    plt.plot(rain, label='rain')
    # plt.plot(wind, '.', label='wind')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plot_weather()
