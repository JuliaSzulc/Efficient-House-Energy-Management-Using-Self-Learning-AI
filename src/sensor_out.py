"""This module provides class for outside sensors

These sensors are meant to reflect current world weather almost exactly.
They are also passing these information forward to the house.

"""
from collections import OrderedDict


class OutsideSensor:
    """Weather sensor on the outside of the house"""

    def __init__(self, house_listener):
        self.daytime = None

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
        """Updates object with info from subject and passes info forward"""
        self.daytime = daytime
        self.weather = weather
        self.house_listener.update(self.get_info())

    def get_info(self):
        """Collect weather info from sensor

        Returns:
            sensor_info(dict): weather information
        """

        # actual_temp is calculated by the formula given in documentation
        sensor_info = OrderedDict({
            'daytime': self.daytime,
            'actual_temp': self.weather['temperature'],
            'light': self.weather['light'],
            'clouds': self.weather['clouds'],
            'rain': self.weather['rain'],
            'wind': self.weather['wind']
        })

        return sensor_info
