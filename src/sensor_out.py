"""This module provides class for outside sensors

These sensors are meant to reflect current world weather almost exactly.
They are also passing these information forward to the house.

"""
from collections import OrderedDict


class OutsideSensor:
    """Weather sensor on the outside of the house"""

    def __init__(self, house_listener):
        """
        Args:
            house_listener(House): House object that has the update() method
                                   implemented
        """
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

        sensor_info = OrderedDict({
            'Daytime': self.daytime,
            'Outside Temp': self.weather['temperature'],
            'Outside Light': self.weather['light'],
            'Clouds': self.weather['clouds'],
            'Rain': self.weather['rain'],
            'Wind': self.weather['wind']
        })

        return sensor_info
