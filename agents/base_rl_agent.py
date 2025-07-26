import numpy as np

class BaseRLAgent:
    """
    Abstract RL agent interface for poker environments.
    Extend this for specific RL algorithms (e.g., PPO, DQN, etc).
    """

    def __init__(self, name="RLAgent"):
        self.name = name
        self.model = None  # RL model (e.g., SB3 policy)
        self.training = True

    def act(self, observation):
        """
        Given an observation, return an action.
        Override this in subclasses to use your RL model.
        """
        raise NotImplementedError("act() must be implemented by subclass.")

    def learn(self, *args, **kwargs):
        """
        Train the agent on collected experience.
        Override in subclasses.
        """
        raise NotImplementedError("learn() must be implemented by subclass.")

    def reset(self):
        """
        Reset any internal state at the start of a new episode/hand.
        """
        pass

    def save(self, path):
        """
        Save the agent's model/weights.
        """
        if self.model:
            self.model.save(path)

    def load(self, path):
        """
        Load the agent's model/weights.
        """
        if self.model:
            self.model.load(path)