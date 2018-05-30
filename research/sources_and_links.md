# RL-for-decission-process
## Sources and links

### Environment modelling

- [Daylight illuminance info] (https://greenbusinesslight.com/resources/lighting-lux-lumens-watts/)
- [Wind chill formula] (https://pl.wikipedia.org/wiki/Temperatura_odczuwalna#Model_2_%E2%80%93_temperatura-wiatr)
- [Optimal illuminance levels] (https://panasonic.net/ecosolutions/lighting/technology/knowledge/03/)

### Python

- [Codewars] (https://www.codewars.com/)

Po zrozumieniu Pythona w teorii, warto poćwiczyć w praktyce na różnego rodzaju zadaniach. Polecamy też zadania z adventofcode.com/2017 ;)

- [Dive Into Python 3] (http://www.diveintopython3.net/)

Darmowy PDF - dobra, konkretna książka, przerobić pierwsze kilka rozdziałów

- [90% of Python in 90 Minutes](https://www.slideshare.net/MattHarrison4/learn-90)

### RL
- [RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- [Basic notes to the lectures](https://gist.github.com/mouradmourafiq/b78d9b74545e7c75db48dd9e45abfe5c)

Tak jak mówiłem te notatki to fajny skrypt zagadnien z wykładu, wszystko zebrane wraz z minimalną, ale przydatną intuicją do zagadnień.

- [Repo with RL Algorithms implementations (epic!)](https://github.com/dennybritz/reinforcement-learning)

Wiele algorytmów RL na licencji pozwalającej na commercial use, te z wykładu i też bardziej zaawansowane, z użyciem deep learning między innymi. Dobrze jest sobie co jakiś czas analizować kod algorytmów, które poznało się na wykładzie.

- [Awesome RL - zbiór wielu wielu jakościowych źródeł o RL](https://github.com/aikorea/awesome-rl)

- [RL Intro Holy Grail Book](http://incompleteideas.net/book/bookdraft2018jan1.pdf)

Do poczytania po wykładzie lub jako uzupełnienie, wykład opiera się silnie na tej książce. Co warto sprawdzić - na końcach tematów/rozdziałów są często zadania i teoretyczne i praktyczne.

- [A. Karpathy (znana osoba w ML) na temat RL](http://karpathy.github.io/2016/05/31/rl/)

Blog post, trochę innych intuicji do różnego rodzaju metod, jego historia (od zera do stażu w deepmind) itd.

- [OpenAI Gym repo](https://github.com/openai/gym)

Środowiska do nauki agentów, od prostych na których można ćwiczyć algorytmy RL, po gry atari

#### Q-Learning Double Q-Learning, Deep Double Q-Learning

- [Q-Learning - jak i dlaczego działa? (Jeżeli rozumiesz, to powinieneś umieć wytłumaczyć czym to się różni od SARSA)](https://www.quora.com/How-does-Q-learning-work-1)

- [Deep Double Q Learning - Oryginalny paper](https://arxiv.org/pdf/1509.06461.pdf)

- [Reddit thread, w którym twórca tłumaczy działanie i wyjaśnia motywację](https://www.reddit.com/r/MachineLearning/comments/57ec9z/discussion_is_my_understanding_of_double/)

- [Idealne wytłumaczenie QL / DQL wraz z pseudokodem](http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/)

### Neural Networks

- [Backpropagation algorithm explained + more](http://colah.github.io/posts/2015-08-Backprop/)

Mega polecany blog post na temat działania algorytmu backpropagation, który drastycznie zmniejsza czas trenowania sieci, może pod kątem projektu nie jest to najważniejsze, ale pewnie użyjemy sieci prędzej czy później, a to podstawa ich działania. 

- [Sieć neuronowa od zera z wykorzystaniem Numpy](https://github.com/dennybritz/nn-from-scratch)

streszczenie teorii i kod podstawowej sieci neuronowej z jedną ukrytą warstwą i algorytmem Batch Gradient Descent, warto przerobić, napisać samemu, dobre ćwiczenie żeby zrozumieć zapisy macierzowe z Numpy, bo na początku są dosyć zawiłe.

- [3Blue1Brown - wprowadzenie do sieci neuronowych i Gradient Descent/Backprop](https://www.youtube.com/watch?v=aircAruvnKk)

Mistrzowski kanał, filmy z mega dobrymi wizualizacjami. (są chyba 3 części)

### PyTorch

- [Dokumentacja PyTorch] (http://pytorch.org/tutorials/#)

Rzekomo jedno z najlepszych miejsc do nauki, dobra jakość dokumentacji i tutoriale (w tym taki podstawowy tutorial dla nowych osób [tutaj](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html))

- [PyTorch Tutorial Github repo](https://github.com/yunjey/pytorch-tutorial)

Repo z kodem wielu podstawowych rzeczy, które można zrobić z PyTorchem, prawdopodobnie najlepsze źródło do treningu, jak już się wie o co chodzi w teorii.

- [PyTorch ZeroToAll](https://drive.google.com/drive/folders/0B41Zbb4c8HVyUndGdGdJSXd5d3M)

Slajdy do DL w PyTorch z minimalnym wprowadzeniem teoretycznym. Średnia jakość, ale wiedzy (i kodu) całkiem sporo.

### Papers

- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
