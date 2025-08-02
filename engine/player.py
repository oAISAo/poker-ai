# poker-ai/engine/player.py


from typing import Optional
from typing import Optional, List
from agents.base_agent import BaseAgent
from engine.cards import Card

class Player:
    def __init__(self, name: str, stack: int = 1000, is_human: bool = False):
        self.name = name
        self.stack = stack
        self.hole_cards: List[Card] = []
        self.current_bet = 0
        self.in_hand = True  # True if player hasn't folded
        self.is_human = is_human
        self.all_in = False  # Track if player is all-in
        self.total_contributed = 0  # Track total chips put in pot this hand
        self.agent = None  # type: Optional[BaseAgent]

    def deal_hole_cards(self, cards):
        if len(cards) != 2:
            raise ValueError("Texas Hold'em players get exactly 2 hole cards.")
        self.hole_cards = cards

    def bet_chips(self, amount, suppress_log=False):
        old_stack = self.stack
        old_current_bet = self.current_bet
        old_total_contributed = self.total_contributed
        print(f"[PLAYER bet_chips] {self.name} called bet_chips({amount}, suppress_log={suppress_log})")
        actual_bet = min(self.stack, amount)
        self.stack -= actual_bet
        self.current_bet += actual_bet
        self.total_contributed += actual_bet  # Track total for side pots
        if self.stack == 0:
            self.all_in = True  # Player is all-in if no chips left
        print(f"[PLAYER bet_chips] {self.name}: amount={amount}, stack: {old_stack}->{self.stack}, current_bet: {old_current_bet}->{self.current_bet}, total_contributed: {old_total_contributed}->{self.total_contributed}")
        if not suppress_log:
            print(f"[PLAYER] {self.name} bets {actual_bet}. Remaining stack: {self.stack}")
        return actual_bet

    def post_ante(self, amount, suppress_log=False):
        """Post ante - doesn't count toward current_bet, only total_contributed and pot"""
        old_stack = self.stack
        old_total_contributed = self.total_contributed
        print(f"[PLAYER post_ante] {self.name} called post_ante({amount}, suppress_log={suppress_log})")
        actual_ante = min(self.stack, amount)
        self.stack -= actual_ante
        # NOTE: Ante does NOT count toward current_bet in Texas Hold'em
        # self.current_bet += actual_ante  # <-- This line is intentionally commented out
        self.total_contributed += actual_ante  # Track total for side pots
        if self.stack == 0:
            self.all_in = True  # Player is all-in if no chips left
        print(f"[PLAYER post_ante] {self.name}: amount={amount}, stack: {old_stack}->{self.stack}, total_contributed: {old_total_contributed}->{self.total_contributed}")
        if not suppress_log:
            print(f"[PLAYER] {self.name} posts ante of {actual_ante}. Remaining stack: {self.stack}")
        return actual_ante

    def fold(self):
        self.in_hand = False
        print(f"[PLAYER] {self.name} folds.")

    def reset_for_new_hand(self):
        self.hole_cards = []
        self.current_bet = 0
        self.in_hand = True
        self.all_in = False  # Reset all-in status at the start of a new hand
        self.total_contributed = 0  # Reset for new hand

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
