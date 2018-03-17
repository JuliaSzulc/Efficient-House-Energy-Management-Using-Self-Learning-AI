# Requirements to run: pymongo (python -m pip install pymongo), mongodb installed and run
from pymongo import MongoClient

client = MongoClient()
client = MongoClient('localhost', 27017)  # host and port of running mongodb instance

db = client['RL_results_database']

post = {"current_date": "1",  # from world
        "daytime": "",  # from world
        "weather": "",  # from world, elements: 'temp','sun','clouds','rain','wind'
        "grid_cost": "",  # from house
        "current_settings": "",  # from house, elements: 'energy_src', 'cooling_lvl', 'heating_lvl', 'light_lvl', 'curtains_lvl'
        "battery": "",  # from house, elements: current, max(?)
        "inside_sensors": "",  # from house, dict of sensors with 'name', 'temperature', 'light'
        "pv_absorption": "",  # from house
        "user_requests": "",  # from house, dict of settings, which contains 'name', 'temp_desired', 'temp_epsilon', 'light_desired', 'light_epsilon'
        "action_taken": "",
        "reward": ""
        }

posts = db.results
post_id = posts.insert_one(post).inserted_id

