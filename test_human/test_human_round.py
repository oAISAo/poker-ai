# test_human/test_human_round.py
# run: python test_human/test_human_round.py

from engine.game import PokerGame, GameMode
from engine.player import Player
import random

def main():
    print("=== Poker AI Interactive Test ===")
    random.seed(42)  # Set seed for consistency

    # Create players: You (human), 2 AIs
    players = [
        Player("You", stack=1000, is_human=True),
        Player("Bot_1", stack=1000, is_human=False),
        Player("Bot_2", stack=1000, is_human=False)
    ]

    game = PokerGame(players, game_mode=GameMode.HUMAN_VS_AI)

    hand_count = 0
    try:
        while True:
            hand_count += 1
            print(f"\n\n======== Hand #{hand_count} ========")
            game.play_hand()

            print("\nStacks after hand:")
            for p in players:
                print(f"{p.name}: {p.stack} chips")

            # Ask if user wants to play another hand
            choice = input("\nPlay another hand? (y/n): ").strip().lower()
            if choice != 'y':
                break
    except KeyboardInterrupt:
        print("\nGame interrupted.")

    print("\nThanks for testing!")

if __name__ == "__main__":
    main()
