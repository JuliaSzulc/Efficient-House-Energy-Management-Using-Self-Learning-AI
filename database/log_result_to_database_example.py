# Requirements to run: pymongo (python -m pip install pymongo), mongodb installed and run
import subprocess
from pymongo import MongoClient

client = MongoClient()
client = MongoClient('localhost', 27017)  # host and port of running mongodb instance

db = client['RL_results_database']


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])


post = {"git_revision_hash": get_git_revision_short_hash(),  # get current commit hash
        "start_date": "",  # experiment start date
        "end_date": "",  # experiment end date
        "weather_settings": "",
        "inside_sensors": "",  # from house, dict of sensors with 'name'
        "actions_available": "",
        "agent_learned_params": "",
        "reward_from_episodes": ""  # array of rewards from all episodes
        }

posts = db.results
post_id = posts.insert_one(post).inserted_id

