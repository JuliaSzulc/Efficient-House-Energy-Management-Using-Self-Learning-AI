class OutsideSensor:
    """Weather sensor on the outside of the house"""

    def __init__(self):
        self.daytime = None
        self.weather = None

    def update(self, weather, daytime, **kwargs):
        self.daytime = daytime
        self.weather = weather
    
    def get_info(self):
        """Collect weather info from sensor

        Returns:
            info (type?) : weather informations

        """
        # TODO: do sth with weather

        return self.weather
