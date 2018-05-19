"""This is the main executable module for the project.

Performs training of new - or loaded - model of the RL agent and provides
logging, plotting and saving options. If 'manual' option is specified, there
is no training.

You can run this file with additional arguments:
    manual   - enables manual test mode
    stats    - print more stats about each episode
    save     - saves experiment results
    load     - load agent model from file before training
    plot     - make plot of the total rewards of each episode (default True)
    safemode - log total reward to rewards.log after each episode
    quiet    - disable printing total rewards to console

"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from manual_test import ManualTestTerminal
from environment import HouseEnergyEnvironment
from agent import Agent
from shutil import copyfile


def main():
    # TODO: write a "just run it" test, to check consistency
    # TODO: posprzątać te flagi i 'if-y', za dużo tego, wywalić te flagi z =True,
    # wywalić safemode, wywalić manual test.
    save_experiment = False
    run_manual_tests = False
    print_stats = True
    make_total_reward_plot = True
    load_agent_model = False
    safemode = False
    quiet = False

    if 'manual' in sys.argv:
        run_manual_tests = True
    if 'stats' in sys.argv:
        print_stats = True
    if 'save' in sys.argv:
        save_experiment = True
    if 'load' in sys.argv:
        load_agent_model = True
    if 'plot=False' in sys.argv:
        make_total_reward_plot = False
    if 'plot=True' in sys.argv:
        make_total_reward_plot = True
    if 'safemode' in sys.argv:
        safemode = True
    if 'quiet' in sys.argv:
        quiet = True

    # TODO: Manual testing as a separate script
    if run_manual_tests:
        tests = ManualTestTerminal()
        tests.manual_testing()
        return

    # --- initialization ---
    env = HouseEnergyEnvironment(collect_stats=print_stats)
    agent = Agent(env=env)

    model_id = -1
    if load_agent_model:
        model_id = input('Enter model number to load:\n')
        load_model(agent, model_id)

    open('rewards.log', 'w').close()  # reset rewards log


    add_path = ''
    if 'tests' in os.getcwd():
        add_path = '../'
    with open(add_path + '../configuration.json') as config_file:
        config = json.load(config_file)

    training_episodes = config['main']['training_episodes']
    # --- learning ---
    rewards = []
    for i in range(training_episodes):
        t_reward = agent.run()
        rewards.append(t_reward)

        if safemode:
            with open("rewards.log", "a") as logfile:
                logfile.write("{}\n".format(t_reward))

        if not quiet:
            print("episode {} / {} | Reward: {}".format(i, training_episodes,
                                                        t_reward))
            if print_stats:
                print_episode_stats(agent.get_episode_stats(),
                                    env.get_episode_stats())

    if make_total_reward_plot:
        plot_total_rewards(rewards, training_episodes, avg=10)

    if save_experiment:
        config =\
            save_model_info(model_id, agent.q_network,
                            rewards, load_agent_model)

    for param, val in config['agent'].items():
        print(param, val)


def plot_total_rewards(rewards, num_episodes, avg=10): # pragma: no cover
    # Note: avg has to be a divisor of num_episodes
    avg_rewards = []
    for i in range(num_episodes // (avg or 1)):
        avg_rewards.append(np.mean(rewards[avg * i: avg * (i + 1)]))

    plt.plot(avg_rewards)
    plt.show()


def print_episode_stats(agent_stats, env_stats): # pragma: no cover
    print("------------------------------------------------------------------")
    for k, v in agent_stats.items():
        try:
            name = v[0]
            value = v[1]
            print("{:30} = {:20} ({:.3f})".format(k, name, value))
        except TypeError:
            print("{:30} = {:.3f}".format(k, v))

    for k, v in env_stats.items():
        print("{:30} = {: .1f} %".format(k, v))
    print("==================================================================")

# FIXME move loading and saving to a separate class
def load_model(agent, model_id):
    # TODO: write some tests
    """
    Loads the given model to the Agent's network fields.

    Args:
        agent(Agent): agent object which we want to load
        model_id(number): model's number used to find the corresponding file

    """

    try:
        agent.load_config('saved_models/model_{}/configuration.json'.
                          format(model_id))
    except FileNotFoundError:
        print("Loading model failed. No model with given index, or no\
              configuration file")
        sys.exit()

    agent.load_network_model('saved_models/model_{}/network.pt'.
                             format(model_id))


def save_model_info(model_id, model, rewards, model_was_loaded=False):
    # TODO: write some tests
    """
    Method saves the model, configuration file  and training rewards to a files
    in the saved_models/{model_id} directory.

    Args:
        model_id(number): id of the model (should be -1 if it's a new model)
        model(torch.nn.Net): neural network torch model (q_network)
        rewards(list): list of total rewards for each episode
        model_was_loaded(bool): determine if model was loaded or created
    """

    # create new directory with incremented id
    new_index = 0
    while True:
        if not os.path.exists('saved_models/model_{}'.format(new_index)):
            os.makedirs('saved_models/model_{}'.format(new_index))
            break
        new_index += 1

    # copy old rewards log to append new if model was loaded
    # model_was_loaded = (model_id != -1) and os.path.isfile(
        # 'saved_models/model_{}/rewards.log'.format(model_id))
    if model_was_loaded:
        copyfile(
            'saved_models/model_{}/rewards.log'.format(model_id),
            'saved_models/model_{}/rewards.log'.format(new_index))

    #  --- save new data
    # model
    torch.save(model.state_dict(), 'saved_models/model_{}/network.pt'
               .format(new_index))

    # rewards log
    logfile = open("saved_models/model_{}/rewards.log".format(new_index), "a")
    for reward in rewards:
        logfile.write("{}\n".format(reward))
    logfile.close()

    # config
    if model_was_loaded:
        config = "saved_models/model_{}/configuration.json".format(model_id)
    else:
        add_path = ''
        if 'tests' in os.getcwd():
            add_path = '../'
        with open(add_path + '../configuration.json') as config_file:
            config = json.load(config_file)

    copyfile(
        config,
        "saved_models/model_{}/configuration.json".format(new_index))

    # rewards chart
    rewards = []
    for line in open('saved_models/model_{}/rewards.log'.format(new_index), 'r'):
        values = [float(s) for s in line.split()]
        rewards.append(values)
    avg_rewards = []
    for i in range(len(rewards) // (10 or 1)):
        avg_rewards.append(np.mean(rewards[10 * i: 10 * (i + 1)]))
    plt.plot(avg_rewards)
    plt.savefig('saved_models/model_{}/learning_plot.png'.format(new_index))

    with open(
        "saved_models/model_{}/configuration.json".format(new_index)
    ) as config_file:
        CONFIG_AGENT = json.load(config_file)
    return CONFIG_AGENT


if __name__ == "__main__": # pragma: no cover
    main()
