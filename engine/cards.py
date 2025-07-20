# poker-ai/engine/cards.py

import random

# Ranks and suits for standard 52-card deck
RANKS = "2 3 4 5 6 7 8 9 T J Q K A".split()
SUITS = "♠ ♥ ♦ ♣".split()  # Unicode suits for display

class Card:
    def __init__(self, rank, suit):
        if rank not in RANKS:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        self.rank = rank
        self.suit = suit

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        return str(self)

    def to_tuple(self):
        return (self.rank, self.suit)

class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, n=1):
        if n <= 0:
            raise ValueError("Number of cards to draw must be positive")
        if n > len(self.cards):
            raise ValueError("Not enough cards left in the deck")
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn


    def reset(self):
        self.__init__()

    def __len__(self):
        return len(self.cards)

    def __str__(self):
        return f"Deck with {len(self.cards)} cards."

