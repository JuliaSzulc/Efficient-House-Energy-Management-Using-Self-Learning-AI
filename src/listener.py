class Listener:
    """Base class receiving updates from subject"""

    def __init__(self, subject):
        self.subject = subject
        subject.register(self)

        self.daytime = None
        self.weather = []

    def update(self, new_daytime, new_weather):
        # this method is called by subject
        self.daytime = new_daytime
        self.weather = new_weather

