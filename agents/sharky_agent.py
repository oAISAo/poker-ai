import random
from agents.rl_agent import RLAgent
from utils.logger import logger

class SharkyAgent(RLAgent):
    """
    A dummy reinforcement learning agent that randomly selects legal actions.
    This is a placeholder implementation to test the RLAgent interface.
    """

    def __init__(self, name="Sharky"):
        super().__init__(name)

    def train(self, env, timesteps: int):
        """
        Dummy train method — logs training but performs no learning.
        """
        logger.info(f"[{self.name}] Starting dummy training for {timesteps} timesteps...")
        for step in range(1, timesteps + 1):
            if step % max(1, timesteps // 10) == 0:
                logger.info(f"[{self.name}] Training step {step}/{timesteps}")
        logger.info(f"[{self.name}] Dummy training complete.")
        self.is_trained = True

    def get_action(self, state):
        """
        Select a random legal action from the provided state.
        Expects 'state' to contain a key 'legal_actions'.
        """
        legal_actions = state.get("legal_actions", [])
        if not legal_actions:
            logger.warning(f"[{self.name}] No legal actions available.")
            return None
        action = random.choice(legal_actions)
        logger.debug(f"[{self.name}] Chose action: {action}")
        return action
