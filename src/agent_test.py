from src.agent import Agent
import numpy as np


class TestEnv:

    def __init__(self):
        self.state = np.array([0.1, 0.2, 0.3, 0.4, 0.6])
        self.action_space = ['action1', 'action2', 'action3', 'actionterminal']
        pass

    def reset(self):
        return self.state

    def get_actions(self):
        return self.action_space

    def step(self, action):
        return self.state, -5, (action == 'actionterminal')


env = TestEnv()
agent = Agent(env)
agent.memory = [(np.array([0.1, 0.2, 0.3, 0.4, 0.6]), 1, -5, np.array([0.1, 0.2, 0.3, 0.4, 0.6]), False),
                (np.array([0.1, 0.2, 0.3, 0.4, 0.6]), 1, -5, np.array([0.1, 0.2, 0.3, 0.4, 0.6]), False),
                (np.array([0.1, 0.2, 0.3, 0.4, 0.6]), 1, -5, np.array([0.1, 0.2, 0.3, 0.4, 0.6]), False),
                (np.array([0.1, 0.2, 0.3, 0.4, 0.6]), 1, -5, np.array([0.1, 0.2, 0.3, 0.4, 0.6]), False),
                ]
agent.batch_size = 4
agent.run()
