# poker-ai/engine/player.py

from engine.cards import Card

class Player:
    def __init__(self, name, stack=1000, is_human=False):
        self.name = name
        self.stack = stack
        self.hole_cards = []  # list of Card objects
        self.current_bet = 0
        self.in_hand = True  # True if player hasn't folded
        self.is_human = is_human
        self.all_in = False  # Track if player is all-in

    def deal_hole_cards(self, cards):
        if len(cards) != 2:
            raise ValueError("Texas Hold'em players get exactly 2 hole cards.")
        self.hole_cards = cards

    def bet_chips(self, amount, suppress_log=False):
        actual_bet = min(self.stack, amount)
        self.stack -= actual_bet
        self.current_bet += actual_bet
        if self.stack == 0:
            self.all_in = True  # Player is all-in if no chips left
        if not suppress_log:
            print(f"{self.name} bets {actual_bet}. Remaining stack: {self.stack}")
        return actual_bet

    def fold(self):
        self.in_hand = False
        print(f"{self.name} folds.")

    def reset_for_new_hand(self):
        self.hole_cards = []
        self.current_bet = 0
        self.in_hand = True
        self.all_in = False  # Reset all-in status at the start of a new hand

    def decide_action(self, to_call, community_cards):
        # Simple AI logic for testing:
        if to_call == 0:
            return "check"
        elif to_call <= self.stack:
            return "call"
        else:
            return "fold"

    def __str__(self):
        cards_str = ' '.join(str(card) for card in self.hole_cards) if self.hole_cards else "No cards"
        type_str = "Human" if self.is_human else "AI"
        return f"{type_str} Player {self.name} | Stack: {self.stack} | Cards: {cards_str} | In hand: {self.in_hand}"
