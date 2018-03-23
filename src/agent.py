class Agent:
    """Reinforcement Learning agent.

    Agent interacts with the environment, gathering
    information about the reward for his previous actions,
    and observation of state transitions.
    """

    def __init__(self, env):
        self.env = env
        self.actions = self.env.get_actions()
        self.network = None
        self.current_state = None
        self.memory = [] # TODO - czy lista?
        # TODO - algorithm params (gamma, epsilon, batch size etc)
        # network specific params can be moved to the network class,
        # but it might be handy to have all params in one place.
        # Make sure to think this through.
        self.reset()

    def reset(self):
        """Initialize the networks and other parameters"""
        pass

    def run(self):
        """Main agent's function. Performs the deep q-learning algorithm"""
        # TODO: reset musi zwrócić początkowy stan środowiska
        self.current_state = self.env.reset()
        total_reward = 0
        terminal_state = False
        while not terminal_state:
            action = self._get_next_action(self.current_state)
            next_state, reward, terminal_state = self.env.step(action)
            self.memory.append((self.current_state, action, reward,
                                next_state, terminal_state))
            self.current_state = next_state
            total_reward += reward
            self._train()

        return total_reward

    def _train(self):
        """
        Trains the underlying network with use of experience memory
        Note: this method does a training *step*, not whole training
        """
        # TODO implement me!
        pass

    def _get_next_action(self, state):
        """
        Returns next action given a state with use of the network
        Note: this should be epsilon-greedy
        """
        # TODO implement me!
        return 1

    def return_model_info(self):
        """
        Not sure what with this one for now.
        Depends on what pytorch uses to save the network models.
        Method should return the network params and all other params
        that we can reproduce the exact configuration later/save it to the db
        """
        # TODO implement me!
        pass
