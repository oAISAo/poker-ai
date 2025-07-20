import random
from engine.cards import Card, Deck
from engine.game import PokerGame
from engine.player import Player
from utils.enums import GameMode


def test_ai_vs_ai_round():
    num_rounds = 10
    initial_stack = 1000
    random.seed(42)  # Seed for reproducibility

    for round_num in range(num_rounds):
        players = [
            Player(name="Alice", stack=initial_stack, is_human=False),
            Player(name="Bob", stack=initial_stack, is_human=False),
        ]

        game = PokerGame(players, game_mode=GameMode.AI_VS_AI)
        initial_total_stack = sum(p.stack for p in players)

        try:
            game.play_hand()
        except Exception as e:
            print(f"Error during game play: {e}")
            raise e  # Re-raise the exception after logging

        final_total_stack = sum(p.stack for p in players)

        # Assert that the total stack size remains constant
        assert final_total_stack == initial_total_stack, "Total stack size changed!"


# Run this test with: pytest test/auto_test_round.py