# Smart house project using reinforcement learning
The project implements the idea of reinforcement learning for decision process using custom learning agent and environment. The aim is to create a working system for a smart house that analyzes the outside and inside data collected by sensors and determines the action to perform considering the user desired values of temperature and light and energy cost. As a result the system should minimize energy consumption in the house and maximize user’s comfort.

## Requirements:
- Python 3.6
- NumPy
- matplotlib
- PyTorch

For details, see [requirements.txt](https://github.com/JuliaSzulc/RL-for-decission-process/tree/master/requirements.txt) (soon)

## Problem and idea:
We wanted to create a system for a smart house that works properly under any base requirements and conditions and results in visible energy saving.  
Here we think about the smart house as an agent with the access to the basic devices in the house (currently it is: energy source, light level, window blinds, air conditioning and heater) as well as the inside and outside sensors. The sensors collect the data (temperature, brightness and solar battery level) between the fixed timeframes.

## Applied solution:
A complex simulation of the outside world, house and weather has been created as a base environment for the agent. The reward is calculated from difference between current conditions and user requests and energy use. Currently implemented algorithm uses Double Deep Q Learning with Prioritized Experience Replay.

## Instructions:
1. For quick run:
```
$ python3 main.py
```
2. For full-screen live simulation:
```
$ python3 main.py simulation.py
```
3. For manual, step-by-step testing:
```
$ python3 main.py manual_test.py
```
You can change the configuration of agent and environment thorugh [configuration.json](https://github.com/JuliaSzulc/RL-for-decission-process/tree/master/configuration.json)

## Repo Guide:
- [**src**](https://github.com/JuliaSzulc/RL-for-decission-process/tree/master/src) - the whole source code + unit tests
- [**documentation**](https://github.com/JuliaSzulc/RL-for-decission-process/tree/master/documentation) - documentation of the project, diagrams and documents for our university course as well as our coding guidelines and list of used sources
- [**meetings**](https://github.com/JuliaSzulc/RL-for-decission-process/tree/master/meetings) - notes and conclusions from our meetings
- [**playground**](https://github.com/JuliaSzulc/RL-for-decission-process/tree/master/plyground) - environments, algorithms and programs that helps us collectively understand some concepts of ML, RL, Python and PyTorch

22 May 2018,
  
Dawid Czarneta  
Jakub Frąckiewicz  
Filip Olszewski  
Michał Popiel  
Julia Szulc
