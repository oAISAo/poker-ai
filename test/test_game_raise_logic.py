import pytest
from engine.game import PokerGame
from engine.player import Player
from engine.raise_validation import RaiseValidationError


def setup_game():
    alice = Player("Alice")
    bob = Player("Bob")
    game = PokerGame([alice, bob], big_blind=50)

    alice.stack = 1000
    bob.stack = 1000

    game.current_player_idx = 0  # note: your code uses current_player_idx, not current_player_index
    game.current_bet = 50
    game.last_raise_amount = 50
    game.pot = 150
    return game, alice, bob


def test_valid_raise():
    game, alice, _ = setup_game()
    game.handle_raise(alice, raise_to=150)  # 50 to call + 100 raise

    assert alice.stack == 850
    assert alice.current_bet == 150
    assert game.current_bet == 150
    assert game.last_raise_amount == 100
    assert game.pot == 300


def test_raise_all_in():
    game, alice, _ = setup_game()
    alice.stack = 200
    game.handle_raise(alice, raise_to=250)

    assert alice.stack == 0
    assert alice.current_bet == 250
    assert game.pot == 400


# def test_raise_too_small_raises_error():
#     game, alice, _ = setup_game()
#     with pytest.raises(RaiseValidationError):
#         game.handle_raise(alice, raise_to=75)  # Only 25 over current_bet


# def test_raise_over_stack_raises_error():
#     game, alice, _ = setup_game()
#     with pytest.raises(RaiseValidationError):
#         game.handle_raise(alice, raise_to=1100)  # Alice only has 1000


# def test_raise_equal_to_call_raises_error():
#     game, alice, _ = setup_game()
#     with pytest.raises(RaiseValidationError):
#         game.handle_raise(alice, raise_to=50)  # Same as current bet
