"""This it the main executable module for the project.

it sets RL-agent along with environment, prints the results to user and
database etc.

You can run this file with additional arguments:
    manual  - enables manual test mode
    save    - saves experiment results

"""
from agent import Agent
from environment import HouseEnergyEnvironment
import matplotlib.pyplot as plt
from manual_test import ManualTestTerminal
import numpy as np
import sys


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

    if 'manual' in sys.argv:
        run_manual_tests = True
    if 'save' in sys.argv:
        save_experiment = True

    if run_manual_tests:
        tests = ManualTestTerminal()
        tests.manual_testing()

    # --- initialization ---
    env = HouseEnergyEnvironment()
    agent = Agent(env=env)
    num_episodes = 10000
    # clear the contents of log file
    open('rewards.log', 'w').close()

    # --- learning ---
    rewards = []
    for i in range(num_episodes):
        t_reward = agent.run()

        # save results in log file, for safety reasons. We don't want to lose
        # all results becouse of misclick or laptop power loss ;)
        # this logfile is added to .gitignore
        with open("rewards.log", "a") as logfile:
            logfile.write("{}\n".format(t_reward))

        rewards.append(t_reward)
        print("episode {} / {} | Reward: {}".format(i, num_episodes, t_reward))
        print_episode_stats(agent.get_episode_stats())

    avg_rewards = []
    avg = 10  # should be a divisor of num_episodes
    for i in range(num_episodes // avg):
        avg_rewards.append(np.mean(rewards[avg * i: avg * (i + 1)]))

    plt.plot(avg_rewards)
    plt.show()

    # --- saving results ---
    # TODO: recover any important info about env, agent etc.
    info = None
    if save_experiment:
        # save to database
        save_to_database(info)


if __name__ == "__main__":
    main()
