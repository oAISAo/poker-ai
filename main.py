from engine.game import PokerGame
from engine.player import Player

def main():
    print("Starting Poker AI full game simulation...\n")

    players = [
        Player("Alice", 1000),
        Player("Bob", 1000),
        Player("Charlie", 1000)
    ]

    game = PokerGame(players)
    game.play_hand()

    for p in players:
        print(p)

if __name__ == "__main__":
    main()
