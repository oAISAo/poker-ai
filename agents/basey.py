import random
from engine.cards import Card
from agents.base_agent import BaseAgent

class Basey(BaseAgent):
    def __init__(self, name="Basey", simulations=1000):
        super().__init__(name)
        self.simulations = simulations

    def estimate_hand_strength(self, hole_cards, community_cards, deck):
        """
        Estimate hand strength via Monte Carlo simulations.
        Returns a float between 0 and 1 indicating win probability.
        """

        wins = 0
        ties = 0
        losses = 0
        total = self.simulations

        # For simplicity, assume 2 opponents max; can extend later.
        opponents = 2

        for _ in range(total):
            deck_copy = deck[:]  # copy of remaining cards
            random.shuffle(deck_copy)

            # Complete community cards if not all dealt yet
            missing_community = 5 - len(community_cards)
            sim_community = community_cards + deck_copy[:missing_community]
            deck_index = missing_community

            # Sample opponent hands
            opp_hands = []
            for _ in range(opponents):
                opp_hand = deck_copy[deck_index:deck_index+2]
                deck_index += 2
                opp_hands.append(opp_hand)

            # Evaluate all hands
            from engine.hand_evaluator import evaluate_hand

            my_rank, _ = evaluate_hand(hole_cards + sim_community)

            opp_ranks = [evaluate_hand(opp_hand + sim_community)[0] for opp_hand in opp_hands]

            if all(my_rank > opp_rank for opp_rank in opp_ranks):
                wins += 1
            elif any(my_rank == opp_rank for opp_rank in opp_ranks):
                ties += 1
            else:
                losses += 1

        return (wins + ties * 0.5) / total

    def get_action(self, game_state, player_state):
        """
        Decide action based on estimated hand strength.

        game_state: dict with current community cards, pot, etc.
        player_state: dict with hole cards, stack, current bet, etc.
        """

        hole_cards = player_state['hole_cards']
        community_cards = game_state['community_cards']
        deck = game_state['deck']  # list of remaining cards as Card objects

        strength = self.estimate_hand_strength(hole_cards, community_cards, deck)

        # Simple thresholds — tune later
        if strength < 0.3:
            return {'action': 'fold'}
        elif strength < 0.6:
            return {'action': 'call'}
        else:
            # Raise a random amount between min_raise and player's stack fraction
            min_raise = game_state.get('min_raise', 20)
            max_raise = min(player_state['stack'], min_raise * 4)
            amount = random.randint(min_raise, max_raise)
            return {'action': 'raise', 'amount': amount}
