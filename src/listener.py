class Listener:
    """Base class receiving updates from subject

    After inheritance remember to override update() method!

    """

    def __init__(self, subject, reference=None):
        if not reference:
            reference = self
        reference.subject = subject
        subject.register(reference)

    def update(self, **kwargs):
        print("Calling update on basic Listener, maybe you forgot to override?")



