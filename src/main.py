from agent import Agent
from environment import HouseEnergyEnvironment


def save_to_database(info):
    """Save all the information about an experiment to db"""
    print(info)
    pass


def main():
    """Run the experiment and save the results if needed"""

    # TODO: properly implement me (after environment and agent get done)
    save_experiment = False

    env = HouseEnergyEnvironment()
    # env.render() ?
    # params = dict() ?
    agent = Agent(env=env)
    # agent.network = load_model("models/model1.xxx") ?
    # TODO: decide if we define number of episodes here and do a loop
    # or move it to agent and call agent.run() only once
    agent.run()

    # after learning
    # recover any important info about env, agent etc.
    info = None
    if save_experiment:
        # save to database 
        save_to_database(info)


if __name__ == "__main__":
    main()
