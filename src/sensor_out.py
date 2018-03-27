class OutsideSensor:
    """Weather sensor on the outside of the house"""

    def __init__(self, house_listener):
        self.daytime = None

        # can't be None because in the tests environment calls get_info without update
        self.weather = {
            'temp': 12,
            'sun': 0,
            'light': 0,
            'clouds': 0,
            'rain': 0,
            'wind': 0
        }
        
        self.house_listener = house_listener

    def update(self, weather, daytime):
        self.daytime = daytime
        self.weather = weather
        self.house_listener.update(self.get_info())

    def get_info(self):
        """Collect weather info from sensor

        Returns:
            sensor_info (dict) : weather information

        """

        sensor_info = {
            'daytime': self.daytime,
            'wind_chill': 0.045 * (5.27**0.5 + 10.45 - 0.28 *
                self.weather['wind']) * (self.weather['temp'] - 33) + 33,
            'light': self.weather['sun'],
            'illumination': self.weather['sun'] * 25000, # 25k lux is maximum illumination of the ambient daylight
            'clouds': self.weather['clouds'],
            'rain': self.weather['rain'],
            'wind': self.weather['wind']
        }

        return sensor_info
