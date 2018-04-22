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


def save_to_database(info):
    """Save all the information about an experiment to db"""
    # TODO: implement me!
    print(info)


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
    if load_agent_network:
        model_number = input('Enter model number to load\n'
                             '(e.g. to load network_0 enter 0 etc.)\n')
        agent.load_model_info(model_number)

    num_episodes = 1000

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
    # TODO: database module
    if save_experiment:
        # save to database
        # save_to_database(info)
        # for that moment save to file with method below
        agent.save_model_info()


if __name__ == "__main__":
    main()
