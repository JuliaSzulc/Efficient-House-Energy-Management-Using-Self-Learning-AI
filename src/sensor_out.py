from listener import Listener

# TODO: sensor może przekazywać bezpośrednio pogodę(wtedy jest tylko nakładką 
# na gołego listenera), ale mógłby też wprowadzać np. szumy lub wagi, 
# w zależności od tego z której strony domu stoi itd. Do przemyślenia

class OutsideSensor(Listener):
    def __init__(self, subject):
        super().__init__(subject)

