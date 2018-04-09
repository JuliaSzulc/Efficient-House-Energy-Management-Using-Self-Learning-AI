"""This it the main executable module for the project.

it sets RL-agent along with environment, prints the resultes to user and
database etc.

"""
from agent import Agent
from environment import HouseEnergyEnvironment
import matplotlib.pyplot as plt
from manual_test import ManualTestTerminal
import numpy as np

def save_to_database(info):
    """Save all the information about an experiment to db"""
    print(info)


def main():
    """Run the experiment and save the results if needed"""

    # TODO: properly implement me (after environment and agent get done)
    save_experiment = False
    run_manual_tests = False

    # start manual tests
    if run_manual_tests:
        tests = ManualTestTerminal()
        tests.manual_testing()

    env = HouseEnergyEnvironment()
    # env.render() ?
    # params = dict() ?
    agent = Agent(env=env)
    # agent.network = load_model("models/model1.xxx") ?

    rewards = []
    num_episodes = 10000
    for i in range(num_episodes):
        t_reward = agent.run()
        rewards.append(t_reward)
        print("episode {} / {} | Reward: {}".format(i, num_episodes, t_reward))

    avg_rewards = []
    for i in range(100):
        avg_rewards.append(np.mean(rewards[100 * i: 100 * (i+1)]))

    plt.plot(avg_rewards)
    plt.show()

    # after learning
    # recover any important info about env, agent etc.
    info = None
    if save_experiment:
        # save to database
        save_to_database(info)


if __name__ == "__main__":
    main()
