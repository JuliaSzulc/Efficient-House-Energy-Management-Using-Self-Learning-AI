![coverage](https://img.shields.io/badge/coverage-94%25-green.svg)

![cat](https://i.pinimg.com/originals/f3/cc/f7/f3ccf75f1883faea777a00f351c5dc60.gif)
# Smart house project using reinforcement learning :seedling:
The project implements the idea of reinforcement learning for decision process using custom learning agent and environment. The aim is to create a working system for a smart house that analyzes the outside and inside data collected by sensors and determines the action to perform considering the user desired values of temperature and light and energy cost. As a result the system should minimize energy consumption in the house and maximize user’s comfort.    
  
## Problem and idea:  
We wanted to create a system for a smart house that works properly under any base requirements and conditions and results in visible energy saving.
Here we think about the smart house as an agent with the access to the basic devices in the house (currently it is: energy source, light level, window blinds, air conditioning and heater) as well as the inside and outside sensors. The sensors collect the data (temperature, brightness and solar battery level) between the fixed timeframes.

## Applied solution:
A complex simulation of the outside world, house and weather has been created as a base environment for the agent. The reward is calculated from difference between current conditions and user requests and energy use. Currently implemented algorithm uses Double Deep Q Learning with Prioritized Experience Replay.  

## Requirements:  
see [requirements.txt](https://github.com/JuliaSzulc/RL-for-decission-process/tree/master/requirements.txt)
  
## Building:
```
$ pip3 install -r requirements.py
```

## Instructions:
1. To run a training session:
```
$ python3 main.py
```
2. For full-screen live simulation:
```
$ python3 simulation.py
```
3. For manual, step-by-step testing:
```
$ python3 manual_test.py
```
You can change the configuration of agent and environment thorugh [configuration.json](https://github.com/JuliaSzulc/RL-for-decission-process/tree/master/configuration.json)

## Repo Guide:
<pre>
<b>.</b>
├── configuration.json       <i> Main configuration file </i>
├── README.md
├── requirements.txt
├── ...
├── <b>documentation</b>            <i> Both technical and business documentation of the project </i>
│   └── ...
├── <b>research</b>                 <i> Our additional research notes </i>
│   └── ...
├── <b>src</b>                      <i> Source code directory </i>
│   ├── main.py              <i> Main script used to train Agent </i>
│   ├── simulation.py        <i> Graphical simulation of Agent at work </i>
│   ├── manual_test.py       <i> Console-based simulation of Agent and Environment</i>
│   ├── profile.sh           <i> Our profiling script</i>
│   ├── ...
│   ├── <b>saved_models</b>         <i> Directory for storing models </i>
│   │   └── ...
│   └── <b>tests</b>                <i> Unit tests directory </i>
│       ├── get_coverage.sh  <i> Our testing script </i>
│       └── ...
└── <b>static</b>                   <i> Directory for storing static files </i>
    ├── <b>fonts</b> 
    └── <b>icons</b> 
 </pre>

For more info please refer to [**documentation**](https://github.com/JuliaSzulc/RL-for-decission-process/tree/master/documentation) of the project, where you can find diagrams and documents for our university course as well as our coding guidelines and list of used sources.
  
22 May 2018,  
  
:bear: Dawid Czarneta  
:tiger2: Jakub Frąckiewicz  
:wolf: Filip Olszewski  
:boar: Michał Popiel  
:cat: Julia Szulc  
