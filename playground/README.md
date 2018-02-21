## Playground

Moja propozycja jest taka, żeby tutaj wrzucać swoje algorytmy, środowiska itd. 

### 1D Path Env

Zacznę od wrzucenia środowiska 1DPathEnv, na którym można poćwiczyć Model-Free Prediction / Model-Free Control.

Zasady - środowisko to ścieżka, można poruszać się w lewo i w prawo, stany są kolejno ponumerowane, tworząc środowisko podajesz jako listę stany terminalne. Wejście do stanu terminalnego zwraca nagrodę 1, inne akcje dają nagrodę 0. 

Użycie - tworzysz, po czym generujesz epizody funkcją *sample_episode*

### Monte Carlo Policy Evaluation (Prediction)

MC_RW_Prediction.py


