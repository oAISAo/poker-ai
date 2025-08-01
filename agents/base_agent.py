from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name="BaseAgent"):
        self.name = name

    @abstractmethod
    def get_action(self, game_state, player_state):
        """
        Decide on an action given the current game and player state.
        

        Parameters:
        - game_state: full info about game, community cards, betting round, pot, etc.
        - player_state: info about the agent's own hand, stack, current bet, etc.

        Returns:
        - action: dict or tuple representing the chosen action (e.g., {'action': 'call'} or {'action': 'raise', 'amount': 100})
        """
        pass

    def __str__(self):
        return f"<Agent: {self.name}>"
