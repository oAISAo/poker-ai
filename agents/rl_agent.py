from abc import ABC, abstractmethod
from utils.logger import logger
from agents.base_agent import BaseAgent

class RLAgent(BaseAgent, ABC):
    """
    Abstract base class for reinforcement learning poker agents.
    Defines the structure for training, action selection, and model management.
    """

    def __init__(self, name="RLAgent"):
        super().__init__(name)
        self.model = None  # Placeholder for the RL model (e.g., PPO, DQN, etc.)
        self.is_trained = False

    @abstractmethod
    def train(self, env, timesteps: int):
        """
        Train the agent on the given environment for a number of timesteps.

        Args:
            env: A training environment instance (e.g., a Gym-like poker environment).
            timesteps (int): The number of steps or episodes to train for.
        """
        raise NotImplementedError("train() must be implemented by subclasses")

    @abstractmethod
    def get_action(self, state):
        """
        Choose an action given the current state of the game.

        Args:
            state: A representation of the current environment/game state.

        Returns:
            An action selected according to the agent's policy/model.
        """
        raise NotImplementedError("get_action() must be implemented by subclasses")

    def save_model(self, path: str):
        """
        Save the agent's trained model to the given file path.

        Args:
            path (str): Destination file path for saving the model.
        """
        if self.model is None:
            logger.warning("No model to save.")
            return

        # Replace with actual model save logic, e.g., self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load a model from a given file path.

        Args:
            path (str): Path to the model file to load.
        """
        # Replace with actual model loading logic, e.g., self.model = ModelClass.load(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
