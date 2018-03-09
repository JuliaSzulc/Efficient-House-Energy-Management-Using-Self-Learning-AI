class Listener:
    def __init__(self, subject):
        self.subject = subject
        subject.register(self)

        self.daytime = None

    def update(self, new_daytime):
        self.daytime = new_daytime
