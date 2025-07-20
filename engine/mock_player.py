# filepath: engine/mock_player.py
from engine.player import Player

class MockPlayer(Player):
    def __init__(self, name, stack):
        super().__init__(name, stack)
        self.is_human = False  # Override to skip input

    def decide_action(self, to_call, community_cards):
        """Very simple AI: check if possible, otherwise call."""
        if to_call == 0:
            return 'check'
        elif to_call <= self.stack:
            return 'call'
        else:
            return 'fold'