import random
from engine.game import PokerGame
from engine.player import Player
from utils.enums import GameMode

def test_multi_hand_session():
    random.seed(123)
    players = [
        Player("Alice", stack=1000, is_human=False),
        Player("Bob", stack=1000, is_human=False),
        Player("Carol", stack=1000, is_human=False)
    ]

    game = PokerGame(players, game_mode=GameMode.AI_VS_AI)
    num_hands = 5

    for hand_num in range(num_hands):
        print(f"\nStarting hand {hand_num + 1}")
        initial_total = sum(p.stack for p in players)

        game.play_hand()

        final_total = sum(p.stack for p in players)
        assert final_total == initial_total, "Total chips should remain constant"

        # Check no player has negative stack
        for p in players:
            assert p.stack >= 0, f"{p.name} has negative stack"

        # Dealer should rotate
        expected_dealer = (hand_num + 1) % len(players)
        assert game.dealer_position == expected_dealer, "Dealer position mismatch"

