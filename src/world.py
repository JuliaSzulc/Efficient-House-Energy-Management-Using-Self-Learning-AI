"""This module provides outside world simulations.

Grouped in World class, these methods play a major role in
HouseEnergyEnvironment. World takes care of weather and time computations, and
sends them forward to all listeners - house and outside sensors.

It should be used inside environment class, unless you want to plot
an example weather graph, than you can run this file as a script.

Information for listeners are the daytime and the weather.
'sun' value is the sun power before calculating with clouds
'light' value is the sun power after this calculation


Note on algorithms:

Most of the weights and biases were determined experimentally.
The main goal was to provide natural-looking weather, not self-corelated, with
nicely varied rain periods, clouds, ocassional storms etc.
In order to achieve this, in some places a modified version of Gillbert-Elliot
channel model was used to simulate constant periods of given weather phenomena.
Please refer to its logic first, in case of being lost in code.

"""
import json
import os
import random
from datetime import datetime, timedelta
from math import sin, pi
from tools import truncate


class World:
    """Time and weather computations"""

    def __init__(self, time_step_in_minutes=None, duration_days=1):
        """
        Args:
            time_step_in_minutes(int): how long a single step will last.
            duration_days(int): how many days the episode will run. For endless
                                loop, make it None

        """

        # read config
        add_path = ''
        if 'tests' in os.getcwd():
            add_path = '../'
        with open(add_path + '../configuration.json') as config_file:
            self.config = json.load(config_file)

        self.start_date = datetime(2020, 1, 1, 0, 0, 0)  # arbitrary start date
        self.stop_date = None
        if duration_days:
            self.stop_date = self.start_date + timedelta(days=duration_days)

        # "real", interpolated values
        self.current_date = self.start_date
        self.daytime = None
        self.time_step_in_minutes = self.config['env']['timestep_in_minutes']
        if time_step_in_minutes:
            self.time_step_in_minutes = time_step_in_minutes

        self.time_step = timedelta(minutes=self.time_step_in_minutes)

        # base values
        self.base_date = self.current_date
        self.base_step_in_minutes = 5  # base step for weather events.
        self.base_step = timedelta(minutes=self.base_step_in_minutes)
        self.basetime = None

        # compute initial time
        self._compute_daytime()
        self._compute_basetime()

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
        self.sun_amplitude = random.uniform(0.5, 1)

        # other settings
        self.listeners = []

    def register(self, listener):
        """Registers listener on listeners list.

        Args:
            listener(object) - object to be added to self.listeners.
                Should implement update() method.
        """

        self.listeners.append(listener)

    def update_listeners(self):
        """Update all listeners with interpolated weather and current time"""

        for listener in self.listeners:
            try:
                listener.update(daytime=self.daytime,
                                weather=self.int_weather)
            except AttributeError:
                raise AttributeError('listener has unimplemented\
                                     method update')

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

        if self.stop_date and self.current_date >= self.stop_date:
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
        # in place
        now = self.current_date
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.daytime = (now - midnight).seconds / 60

    def _compute_basetime(self):
        # in place
        base_now = self.base_date
        base_midnight = base_now.replace(hour=0, minute=0, second=0,
                                         microsecond=0)
        self.basetime = (base_now - base_midnight).seconds // 60

    def _update_weather(self):
        """Updates all weather factors in proper order"""
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
        """Sun is calculated as a sinus between day start and end hour.

        Amplitude is set once per episode, and varies between <0.5, 1>

        """

        daystart = self.config['env']['day_start']
        dayend = self.config['env']['day_end']
        daylen = dayend - daystart
        sun = 0
        if daystart <= self.basetime <= dayend:
            sun = truncate(sin((self.basetime - daystart) * pi / daylen)) *\
                  self.sun_amplitude

        self.weather['sun'] = sun

    def _calculate_wind(self):
        """Wind is calculated differently when active and inactive.

        We have different propabilities to change between states, and we also
        include previous value as a factor.

        """

        # probabilities to start / stop blowing
        probability_stop = 0.05
        probability_start = 0.02

        # extracted booleans:
        is_blowing = self.weather['wind'] > 0
        going_to_stop = random.uniform(0, 1) < probability_stop
        going_to_start = random.uniform(0, 1) < probability_start

        if is_blowing:
            if going_to_stop:
                self.weather['wind'] = 0
            else:
                old_factor = 0.8
                new_factor = 1 - old_factor

                new_wind = random.betavariate(0.5, 0.5)
                self.weather['wind'] = (self.weather['wind'] * old_factor
                                        + new_wind * new_factor)
        elif going_to_start:
            self.weather['wind'] = random.betavariate(2, 5)

    def _calculate_clouds(self):
        """Clouds are calculated differently when active and inactive.

        We have different propabilities to change between states, and we also
        include previous value and wind as a factor.
        Altough, there is still a small chance of sudden storm or cleariness.

        Clouds have some critical value, and they tend to not pass through it.
        This allows us to simulate mainly moderate weather.

        """

        probability_stop = 0.1
        probability_start = 0.02
        probability_clear_all = 0.004
        probability_storm = 0.002
        probability_pass_critical = 0.9

        # extracted booleans:
        is_cloudy = self.weather['clouds'] > 0
        going_to_stop = random.uniform(0, 1) < probability_stop
        going_to_start = random.uniform(0, 1) < probability_start
        suddenly_clear_all = random.uniform(0, 1) < probability_clear_all
        suddenly_storm = random.uniform(0, 1) < probability_storm
        passing_critical = random.uniform(0, 1) < probability_pass_critical

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
                clouds = (self.weather['clouds'] * old_factor
                          + new_cloud * new_factor)
                # consider influence of wind
                clouds -= self.weather['wind'] * wind_factor
                # go go over 0.4 with the probability of passing critical value
                upper_limit = 0.4
                if passing_critical:
                    upper_limit = 1
                # update clouds
                self.weather['clouds'] = truncate(clouds, 0, upper_limit)

        elif going_to_start:
            # start small, grow tall
            self.weather['clouds'] = random.uniform(0.05, 0.2)

    def _calculate_light(self):
        """Light is calculated from sun and clouds"""
        clouds_factor = 0.7

        self.weather['light'] = truncate(
            self.weather['sun'] - (self.weather['clouds'] * clouds_factor)
        )

    def _calculate_rain(self):
        """Rain is calculated differently when active and inactive.

        We have constant propability to change between states, and we also
        include previous value and clouds as a factor. Note, that there must
        be at least some clouds for rain to fall.

        """
        probability_change_state = 0.05

        # extracted booleans:
        change_state = random.uniform(0, 1) < probability_change_state
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
        """Calculated based on all previous factors, base and tendency.

        Base is the additional bias, very slowly changing, simulating
        long-term weather conditions, such as atmospheric fronts.

        Tendency is a direction for the base - it can be increasing (+1)
        or decreasing (-1). Tendency has a slight chance to change at any
        point in time.

        Besides, temperature is computed including its previous value and some
        hard-coded weights for different factors.

        """

        # make changes in base temperature according to tendency
        change_tendency_probability = 0.001
        base_change_probability = 0.05

        tendency_changing = random.uniform(0, 1) < change_tendency_probability
        base_changing = random.uniform(0, 1) < base_change_probability

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

        # Calculate all temperature factors with their weights.
        # there will be max 8 degrees warmer in a day
        day_heat = 8 * (sin(self.basetime / 230 + 300) / 2 + 0.5)
        sun_heat = 2 * self.weather['light']
        wind_chill = 5 * self.weather['wind']
        rain_chill = 3 * self.weather['rain']

        new_temp = (self.base_temperature
                    + day_heat
                    + sun_heat
                    - wind_chill
                    - rain_chill)

        old_factor = 0.3
        new_factor = 1 - old_factor
        self.weather['temperature'] = truncate(
            old_factor * self.weather['temperature']
            + new_factor * new_temp,
            -20, 40
        )


def plot_weather():  # pragma: no cover
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


if __name__ == '__main__': # pragma: no cover
    import matplotlib.pyplot as plt
    plot_weather()
