"""This module provides class for outside sensors

These sensors are meant to reflect current world weather almost exactly.
They are also passing these informations forward to the house.

"""
from collections import OrderedDict

class OutsideSensor:
    """Weather sensor on the outside of the house"""

    def __init__(self, house_listener):
        self.daytime = None

        # 25k lux is maximum illumination of the ambient daylight
        self.max_illumination = 25000

        self.weather = {
            'temp': None,
            'sun': None,
            'light': None,
            'clouds': None,
            'rain': None,
            'wind': None
        }

        self.house_listener = house_listener

    def update(self, weather, daytime):
        # receives update from subject and passes information forward
        self.daytime = daytime
        self.weather = weather
        self.house_listener.update(self.get_info())

    def get_info(self):
        """Collect weather info from sensor

        Returns:
            sensor_info (dict) : weather information

        """

        # actual_temp is calculated by the formula given in documentation
        sensor_info = OrderedDict({
            'daytime': self.daytime,
            'actual_temp':\
                0.045 * (5.27**0.5 + 10.45 - 0.28 * self.weather['wind']) *\
                (self.weather['temp'] - 33) + 33,
            'light': self.weather['light'],
            'illumination': self.weather['light'] * self.max_illumination,
            'clouds': self.weather['clouds'],
            'rain': self.weather['rain'],
            'wind': self.weather['wind']
        })

        return sensor_info
