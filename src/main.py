"""This it the main executable module for the project.

it sets RL-agent along with environment, prints the results to user and
database etc.

You can run this file with additional arguments:
    manual  - enables manual test mode
    save    - saves experiment results

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from manual_test import ManualTestTerminal
from environment import HouseEnergyEnvironment
from agent import Agent
import os
from shutil import copyfile


def print_episode_stats(dictionary):
    print("-------------------------------------------------------------------")
    for k, v in dictionary.items():
        try:
            name = v[0]
            value = v[1]
            print("{:30} = {:20} ({:.3f})".format(k, name, value))
        except TypeError:
            print("{:30} = {:.3f}".format(k, v))
    print("===================================================================")


def save_model_info(agent, loaded_model_id, data_to_save):
    """
    Method saves all models info to a specific files in
    saved_models directory.

    Args:
        agent(Agent): agent object which we want to load
        loaded_model_id(number): if its different than -1 then its means a model
            was loaded and we needs to recover data from it, so we can
            append new data like reward etc.
        data_to_save(tuple): others data needs to be saved
    """

    # ==========================================================================
    # check existing models and create new with incremented id
    new_index = 0
    while True:
        if not os.path.exists('saved_models/model_{}'.format(new_index)):
            os.makedirs('saved_models/model_{}'.format(new_index))
            break
        new_index += 1

    # ==========================================================================
    # check if any model was loaded before and recover its data

    if loaded_model_id != -1:
        if os.path.isfile(
                'saved_models/model_{}/rewards.log'.format(loaded_model_id)):
            copyfile(
                'saved_models/model_{}/rewards.log'.format(loaded_model_id),
                'saved_models/model_{}/rewards.log'.format(new_index))

    # ==========================================================================
    # save new data part

    agent.save_network_model('saved_models/model_{}/network.pt'
                             .format(new_index))

    logfile = open("saved_models/model_{}/rewards.log".format(new_index), "a")
    for reward in data_to_save[0]:
        logfile.write("{}\n".format(reward))
    logfile.close()

    other_model_data = agent.get_model_info()
    logfile = open("saved_models/model_{}/values.cfg".format(new_index), "w")
    for key, value in other_model_data[0]:
        logfile.write("{:25} {}\n".format(key, value))
    logfile.close()


def load_model_info(agent, model_id):
    """
    Loads the given model to the Agent's network fields.

    Args:
        agent(Agent): agent object which we want to load
        model_id(number): model's number used to find the corresponding file

    """

    try:
        if os.path.isfile(
                'saved_models/model_{}/network.pt'.format(model_id)):
            agent.set_model_info('saved_models/model_{}/network.pt'.
                                 format(model_id))
        else:
            print('[Error] No model with entered index.\n'
                  'Any models have been loaded.\n'
                  'Exiting...')
            raise SystemExit

    except RuntimeError:
        print('[Error] Oops! RuntimeError occurred while loading model.\n'
              'Check if your saved model data is up to date.\n'
              'Maybe it fits different network size?\n'
              'Exiting...')
        raise SystemExit


def main():
    """Run the experiment and save the results if needed"""

    # --- configuration ---
    save_experiment = False
    run_manual_tests = False
    print_stats = False
    make_total_reward_plot = True
    load_agent_network = False
    safemode = False
    quiet = False

    if 'manual' in sys.argv:
        run_manual_tests = True
    if 'stats' in sys.argv:
        print_stats = True
    if 'save' in sys.argv:
        save_experiment = True
    if 'load' in sys.argv:
        load_agent_network = True
    if 'plot=False' in sys.argv:
        make_total_reward_plot = False
    if 'plot=True' in sys.argv:
        make_total_reward_plot = True
    if 'safemode' in sys.argv:
        safemode = True
    if 'quiet' in sys.argv:
        quiet = True

    if run_manual_tests:
        tests = ManualTestTerminal()
        tests.manual_testing()
        return

    # --- initialization ---
    env = HouseEnergyEnvironment()
    agent = Agent(env=env)
    model_id = -1  # needed to recover info which model was loaded
    if load_agent_network:
        model_id = input('Enter model number to load\n'
                         '(e.g. to load network_0 enter 0 etc.)\n')
        load_model_info(agent, model_id)

    num_episodes = 10

    # clear the contents of log file
    open('rewards.log', 'w').close()

    # --- learning ---
    rewards = []
    print("running...")
    for i in range(num_episodes):
        t_reward = agent.run()
        rewards.append(t_reward)

        if safemode:
            with open("rewards.log", "a") as logfile:
                logfile.write("{}\n".format(t_reward))

        if quiet:
            continue

        print("episode {} / {} | Reward: {}".format(i, num_episodes, t_reward))
        if print_stats:
            print_episode_stats(agent.get_episode_stats())

    # --- plotting ---
    if make_total_reward_plot:
        avg_rewards = []
        avg = 10  # has to be a divisor of num_episodes
        for i in range(num_episodes // (avg or 1)):
            avg_rewards.append(np.mean(rewards[avg * i: avg * (i + 1)]))

        plt.plot(avg_rewards)
        plt.show()

    # --- saving results ---
    if save_experiment:
        # !!!
        # If you want to add sth here to be saved pls make sure you
        # did also change saving order of data from data_to_save tuple
        # in save_model_info method.
        # !!!
        data_to_save = rewards, num_episodes
        save_model_info(agent, model_id, data_to_save)


if __name__ == "__main__":
    main()
