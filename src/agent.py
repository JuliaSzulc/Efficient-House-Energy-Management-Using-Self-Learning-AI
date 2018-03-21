class Agent:
    """Reinforcement Learning agent.

    Agent interacts with the environment, gathering
    information about the reward for his previous actions,
    and observation of state transitions.
    """

    def __init__(self, env):
        self.env = env
        self.network = None
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
        # TODO implement me!
        pass

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
        pass

    def return_model_info(self):
        """
        Not sure what with this one for now.
        Depends on what pytorch uses to save the network models.
        Method should return the network params and all other params
        that we can reproduce the exact configuration later/save it to the db
        """
        pass
