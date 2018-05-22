"""This is the main executable module for the project.

Performs training of new - or loaded - model of the RL agent and provides
logging, plotting and saving options. If 'manual' option is specified, there
is no training.

You can change the behaviour details with boolean flags at the beginning
of the main function.

"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from manual_test import ManualTestTerminal
from environment import HouseEnergyEnvironment
from agent import AgentUtils
from agent import Agent
from shutil import copyfile


def main():
    # TODO: write a "just run it" test, to check consistency

    # --- FLAGS ---
    save_experiment  = True
    run_manual_tests = False
    print_stats = True
    make_total_reward_plot = True
    load_agent_model = False
    # ---

    # TODO: Manual testing as a separate script
    if run_manual_tests:
        tests = ManualTestTerminal()
        tests.manual_testing()
        return

    env = HouseEnergyEnvironment(collect_stats=print_stats)
    agent = Agent(env=env)

    model_id = None
    if load_agent_model:
        model_id = input('Enter model number to load:\n')
        AgentUtils.load(agent, model_id)

    with open('../configuration.json') as config_file:
        config = json.load(config_file)

    training_episodes = config['main']['training_episodes']

    # --- learning ---
    rewards = []
    for i in range(training_episodes):
        t_reward = agent.run()
        rewards.append(t_reward)

        print("episode {} / {} | Reward: {}".format(i, training_episodes,
                                                    t_reward))
        if print_stats:
            print_episode_stats(agent.get_episode_stats(),
                                env.get_episode_stats())

    if make_total_reward_plot:
        plot_total_rewards(rewards, training_episodes, avg=10)

    if save_experiment:
        AgentUtils.save(agent, rewards, model_id)


def plot_total_rewards(rewards, num_episodes, avg=10):  # pragma: no cover
    # Note: avg has to be a divisor of num_episodes
    avg_rewards = []
    for i in range(num_episodes // (avg or 1)):
        avg_rewards.append(np.mean(rewards[avg * i: avg * (i + 1)]))

    plt.plot(avg_rewards)
    plt.show()


def print_episode_stats(agent_stats, env_stats):  # pragma: no cover
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


if __name__ == "__main__":  # pragma: no cover
    main()
