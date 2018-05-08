"""This module is used only to train model with floydhub"""
import torch
from environment import HouseEnergyEnvironment
from agent import Agent


def main():
    # --- initialization ---
    env = HouseEnergyEnvironment()
    agent = Agent(env=env)
    num_episodes = 10

    load_model(agent)
    # --- learning ---
    for i in range(num_episodes):
        reward = agent.run()
        print("episode {}, reward {}".format(i, reward))

    save_model_info(agent.q_network)


def load_model(agent):
    try:
        agent.load_network_model('floydhub_gym/network.pt')
        print('model loaded succesfully from gym')
    except FileNotFoundError:
        print('No model in a gym! Starting with new one.')


def save_model_info(model):
    try:
        torch.save(model.state_dict(),
                   '/output/network_trained.pt')
        print('saving successful')
    except:
        print('Something went wrong when saving')


main()
