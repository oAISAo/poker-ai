import pytest
from engine.game import PokerGame
from engine.player import Player
from engine.action_validation import validate_raise, validate_call, ActionValidationError

def setup_game():
    alice = Player("Alice")
    bob = Player("Bob")
    game = PokerGame([alice, bob], big_blind=50)
    alice.stack = 1000
    bob.stack = 1000
    game.current_player_idx = 0
    game.current_bet = 50
    game.last_raise_amount = 50
    game.pot = 150
    return game, alice, bob

# --- handle_call tests ---

def test_call_with_enough_chips():
    game, alice, _ = setup_game()
    # Alice needs to call 50, has 1000
    result = game.handle_call(alice)
    assert alice.stack == 950
    assert alice.current_bet == 50
    assert result["call_amount"] == 50
    assert not result["is_all_in"]

def test_call_all_in_for_less():
    game, alice, _ = setup_game()
    alice.stack = 30
    # Alice needs to call 50, but only has 30 (all-in for less)
    result = game.handle_call(alice)
    assert alice.stack == 0
    assert alice.current_bet == 30
    assert result["call_amount"] == 30
    assert result["is_all_in"]

def test_call_with_exact_stack():
    game, alice, _ = setup_game()
    alice.stack = 50
    # Alice needs to call 50, has exactly 50 (all-in, but not "for less")
    result = game.handle_call(alice)
    assert alice.stack == 0
    assert alice.current_bet == 50
    assert result["call_amount"] == 50
    assert not result["is_all_in"]

def test_call_with_zero_stack_fails():
    game, alice, _ = setup_game()
    alice.stack = 0
    with pytest.raises(ActionValidationError, match="Player has no chips left to call."):
        game.handle_call(alice)

def test_call_with_negative_stack_fails():
    game, alice, _ = setup_game()
    alice.stack = -10
    with pytest.raises(ActionValidationError):
        game.handle_call(alice)

def test_call_with_negative_to_call_fails():
    game, alice, _ = setup_game()
    game.current_bet = -5
    with pytest.raises(ActionValidationError):
        game.handle_call(alice)

def test_call_when_to_call_zero_is_check_not_call():
    game, alice, _ = setup_game()
    game.current_bet = 0
    alice.current_bet = 0
    # Should not be able to "call" when to_call is zero; should be a check
    with pytest.raises(ActionValidationError):
        game.handle_call(alice)

# --- handle_raise tests ---

def test_valid_raise():
    game, alice, _ = setup_game()
    result = game.handle_raise(alice, raise_to=150)  # 50 to call + 100 raise
    assert alice.stack == 850
    assert alice.current_bet == 150
    assert game.current_bet == 150
    assert game.last_raise_amount == 100
    assert game.pot == 300
    assert result["raise_to"] == 150
    assert not result["is_all_in"]

def test_raise_all_in():
    game, alice, _ = setup_game()
    alice.stack = 200
    # Alice's current_bet is 0, so she can only raise_to 200 (all-in)
    result = game.handle_raise(alice, raise_to=alice.current_bet + alice.stack)
    assert alice.stack == 0
    assert alice.current_bet == 200
    assert result["is_all_in"]

def test_minimum_raise():
    game, alice, _ = setup_game()
    result = game.handle_raise(alice, raise_to=100)  # 50 to call + 50 min raise
    assert alice.stack == 900
    assert alice.current_bet == 100
    assert game.current_bet == 100
    assert game.last_raise_amount == 50
    assert not result["is_all_in"]

def test_raise_too_small_raises_error():
    game, alice, _ = setup_game()
    with pytest.raises(ActionValidationError):
        game.handle_raise(alice, raise_to=75)  # Only 25 over current_bet

def test_raise_over_stack_raises_error():
    game, alice, _ = setup_game()
    with pytest.raises(ActionValidationError):
        game.handle_raise(alice, raise_to=1100)  # Alice only has 1000

def test_raise_equal_to_call_raises_error():
    game, alice, _ = setup_game()
    with pytest.raises(ActionValidationError):
        game.handle_raise(alice, raise_to=50)  # Same as current bet

def test_all_in_below_min_raise_allowed():
    game, alice, _ = setup_game()
    alice.stack = 60
    # Alice's current_bet is 0, so she can only raise_to 60 (all-in)
    result = game.handle_raise(alice, raise_to=alice.current_bet + alice.stack)
    assert alice.stack == 0
    assert alice.current_bet == 60
    assert result["is_all_in"]

def test_all_in_not_enough_to_call_raises_error():
    game, alice, _ = setup_game()
    alice.stack = 20
    # to_call = 50, but Alice only has 20, so this is not a valid raise
    with pytest.raises(ActionValidationError):
        game.handle_raise(alice, raise_to=70)  # 70 - 0 = 70, but only 20 in stack

def test_handle_raise_returns_structured_result():
    game, alice, _ = setup_game()
    result = game.handle_raise(alice, raise_to=150)
    assert isinstance(result, dict)
    assert "player" in result
    assert "raise_to" in result
    assert "is_all_in" in result