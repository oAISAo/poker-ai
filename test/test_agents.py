import pytest
from engine.cards import Deck
from agents.basey import Basey

def test_basey_get_action():
    agent = Basey(simulations=100)  # reduce sims for test speed

    deck = Deck()
    hole_cards = deck.draw(2)
    community_cards = deck.draw(3)  # flop

    game_state = {
        'community_cards': community_cards,
        'deck': deck.cards,  # remaining cards
        'min_raise': 20,
        'pot': 100,
        'current_bet': 10,
    }

    player_state = {
        'hole_cards': hole_cards,
        'stack': 1000,
        'current_bet': 10,
    }

    action = agent.get_action(game_state, player_state)
    assert isinstance(action, dict)
    assert 'action' in action
    assert action['action'] in ['fold', 'call', 'raise']

    if action['action'] == 'raise':
        assert 'amount' in action
        amount = int(action['amount'])
        assert 20 <= amount <= player_state['stack']

    print(f"Basey action: {action}")
