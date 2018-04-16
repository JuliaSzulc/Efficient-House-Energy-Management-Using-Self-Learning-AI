"""This module provides outside world simulations.

Grouped in World class, these methods play a major role in
HouseEnergyEnvironment. World takes care of weather and time computations, and
sends them forward to all listeners - house and outside sensors.

It should be used inside environment class only, unless you want to plot
an example weather graph.

"""
from datetime import datetime, timedelta
from math import sin, pi
import random
from random import choices
from tools import truncate


class World:
    """Time and weather computations"""

    def __init__(self, time_step_in_minutes=1, duration_days=1):
        self.start_date = datetime(2020, 1, 1, 0, 0, 0)
        self.stop_date = self.start_date + timedelta(days=duration_days)

        # "real", interpolated values
        self.current_date = self.start_date
        self.daytime = None
        self.time_step_in_minutes = time_step_in_minutes
        self.time_step = timedelta(minutes=time_step_in_minutes)

        # base values
        self.base_date = self.current_date
        self.base_step_in_minutes = 5  # base step for weather events.
        self.base_step = timedelta(minutes=self.base_step_in_minutes)
        self.basetime = None

        self._compute_daytime()
        self._compute_basetime()

        # --- weather part ----
        # sun   -> sun power before calculating with clouds
        # light -> sun power after calculation
        self.weather = {
            'temperature': 12,
            'sun': 0,
            'light': 0,
            'clouds': 0,
            'rain': 0,
            'wind': 0
        }
        self.old_weather = dict(self.weather)
        self.int_weather = dict(self.weather)  # interpolated weather

        self.base_temperature = random.randrange(-10, 30)
        self.tendency = 1  # 1 or -1, the overall temperature will be slowly
        # increasing or decreasing

        self.delta_weather = {
            'temp_delta': 0,
            'sun_delta': 0,
            'light_delta': 0,
        }

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
        """Update all listeners with interpolated weather and current time"""

        for listener in self.listeners:
            try:
                listener.update(daytime=self.daytime,
                                weather=self.int_weather)
            except AttributeError:
                raise AttributeError('listener has unimplemented method update')

    def _interpolate_weather(self):
        linear_factor = (
            self.current_date - (self.base_date - self.base_step)
        ).seconds / (self.base_step_in_minutes * 60)

        for key, value in self.weather.items():
            base = self.old_weather[key]
            delta = value - base
            self.int_weather[key] = base + (delta * linear_factor)

    def step(self):
        """Proceed one step in time, collect info and update listeners

        This method also takes care of the relation between BASE time and
        CURRENT time.

        Returns:
            done(boolean): information if the state after the step is terminal
        """

        if self.current_date >= self.stop_date:
            return True

        self.current_date += self.time_step
        self._compute_daytime()

        while self.current_date > self.base_date:
            self.base_date += self.base_step
            self._compute_basetime()
            self._update_weather()

        self._interpolate_weather()
        self.update_listeners()
        return False

    def _compute_daytime(self):
        now = self.current_date
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.daytime = (now - midnight).seconds / 60

    def _compute_basetime(self):
        base_now = self.base_date
        base_midnight = base_now.replace(hour=0, minute=0, second=0,
                                         microsecond=0)
        self.basetime = (base_now - base_midnight).seconds // 60

    def _update_weather(self):
        for key, value in self.weather.items():
            self.old_weather[key] = value

        # order of methods is important!
        self._calculate_sun()
        self._calculate_wind()
        self._calculate_clouds()
        self._calculate_light()
        self._calculate_rain()
        self._calculate_temperature()

    def _calculate_sun(self):
        daystart = 300  # NOTE: consider moving globals to config file
        dayend = 1140
        daylen = dayend - daystart
        sun = 0
        if self.basetime >= daystart and self.basetime <= dayend:
            sun = truncate(sin((self.basetime - daystart) * pi / daylen))

        self.delta_weather['sun_delta'] = sun - self.weather['sun']
        self.weather['sun'] = sun

    def _calculate_wind(self):
        # propabilities to start / stop blowing
        propability_stop = 0.05
        propability_start = 0.02

        is_blowing = self.weather['wind'] > 0
        going_to_stop = random.uniform(0, 1) < propability_stop
        going_to_start = random.uniform(0, 1) < propability_start

        if is_blowing:
            if going_to_stop:
                self.weather['wind'] = 0
            else:
                old_factor = 0.8
                new_factor = 1 - old_factor

                new_wind = random.betavariate(0.5, 0.5)
                self.weather['wind'] = self.weather['wind'] * old_factor\
                                       + new_wind * new_factor
        elif going_to_start:
            self.weather['wind'] = random.betavariate(2, 5)

    def _calculate_clouds(self):
        propability_stop = 0.1
        propability_start = 0.02
        propability_clear_all = 0.004
        propability_storm = 0.002
        propability_pass_critical = 0.9

        is_cloudy = self.weather['clouds'] > 0
        going_to_stop = random.uniform(0, 1) < propability_stop
        going_to_start = random.uniform(0, 1) < propability_start
        suddenly_clear_all = random.uniform(0, 1) < propability_clear_all
        suddenly_storm = random.uniform(0, 1) < propability_storm
        passing_critical = random.uniform(0, 1) < propability_pass_critical

        if is_cloudy:
            if suddenly_clear_all:
                self.weather['clouds'] = 0

            elif suddenly_storm:
                self.weather['clouds'] = random.uniform(0.8, 1)

            elif going_to_stop:
                self.weather['clouds'] = truncate(self.weather['clouds']
                                                  - random.uniform(0.05, 0.2))
            else:
                old_factor = 0.95
                wind_factor = 0.2
                new_factor = 1 - old_factor

                new_cloud = random.betavariate(5, 1)

                # add new cloud
                clouds = self.weather['clouds'] * old_factor\
                         + new_cloud * new_factor

                # consider influence of wind
                clouds -= self.weather['wind'] * wind_factor

                # go go over 0.4 with the propability of passing critical value
                upper_limit = 0.4
                if passing_critical:
                    upper_limit = 1

                # updaate clouds
                self.weather['clouds'] = truncate(clouds, 0, upper_limit)

        elif going_to_start:
            # start small, grow tall
            self.weather['clouds'] = random.uniform(0.05, 0.2)

    def _calculate_light(self):
        last_light = self.weather['light']
        clouds_factor = 0.7

        self.weather['light'] = truncate(
            self.weather['sun'] - (self.weather['clouds'] * clouds_factor)
        )

        self.delta_weather['light_delta'] = self.weather['light'] - last_light

    def _calculate_rain(self):
        propability_change_state = 0.05

        change_state = random.uniform(0, 1) < propability_change_state
        is_raining = self.weather['rain'] > 0

        if self.weather['clouds'] > 0:
            if is_raining:
                if change_state:
                    self.weather['rain'] = 0

                else:
                    old_factor = 0.3
                    new_factor = 1 - old_factor

                    self.weather['rain'] = truncate(
                        self.weather['rain'] * old_factor
                        + self.weather['clouds'] * new_factor
                    )
            else:
                if change_state and self.weather['clouds'] > 0.4:
                    self.weather['rain'] = truncate(self.weather['clouds']
                                                    * 0.2)
        else:
            self.weather['rain'] = 0

    def _calculate_temperature(self):
        last_temperature = self.weather['temperature']

        # make changes in base temperature according to tendency
        change_tendency_propability = 0.001
        base_change_propability = 0.05

        tendency_changing = random.uniform(0, 1) < change_tendency_propability
        base_changing = random.uniform(0, 1) < base_change_propability

        if tendency_changing:
            self.tendency = -self.tendency

        if base_changing:
            old_factor = 0.9
            new_factor = 1 - old_factor

            new_base_temperature = random.randrange(5, 10) * self.tendency
            self.base_temperature = truncate(
                old_factor * self.base_temperature
                + new_factor * new_base_temperature,
                -20, 40
            )

        # Calculate all temperature factors
        # there will be max 8 degrees warmer in a day
        day_heat = 8 * (sin(self.basetime / 230 + 300) / 2 + 0.5)
        sun_heat = 2 * self.weather['light']
        wind_chill = 5 * self.weather['wind']
        rain_chill = 3 * self.weather['rain']

        new_temp = self.base_temperature\
                   + day_heat\
                   + sun_heat\
                   - wind_chill\
                   - rain_chill\

        old_factor = 0.3
        new_factor = 1 - old_factor
        self.weather['temperature'] = truncate(
            old_factor * self.weather['temperature']
            + new_factor * new_temp,
            -20, 40
        )

        self.delta_weather['temp_delta'] = self.weather['temperature'] \
            - last_temperature


def plot_weather():
    """Plot normalized weather graph in a single episode"""
    temp, sun, light, clouds, rain, wind = [], [], [], [], [], []

    world = World(duration_days=10)
    while not world.step():
        temp.append((world.int_weather['temperature']))
        sun.append(world.int_weather['sun'])
        wind.append(world.int_weather['wind'])
        light.append(world.int_weather['light'])
        clouds.append(world.int_weather['clouds'])
        rain.append(world.int_weather['rain'])

    plt.subplot(411)
    plt.plot(temp, label='temperature')
    plt.legend()

    plt.subplot(412)
    plt.plot(sun, label='sun')
    plt.plot(light, label='light')
    plt.legend()

    plt.subplot(413)
    plt.plot(wind, label='wind')
    plt.plot(clouds, label='clouds')
    plt.legend()

    plt.subplot(414)
    plt.plot(rain, label='rain')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plot_weather()
    # test_base_mechanism()

