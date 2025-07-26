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

# --- handle_fold tests ---

def test_fold_when_in_hand_and_to_call_positive():
    game, alice, _ = setup_game()
    alice.in_hand = True
    game.current_bet = 100
    alice.current_bet = 50
    to_call = game.current_bet - alice.current_bet
    result = game.handle_fold(alice, to_call)
    assert result["can_fold"]
    assert not alice.in_hand

def test_fold_when_in_hand_and_to_call_zero_fails():
    game, alice, _ = setup_game()
    alice.in_hand = True
    game.current_bet = 0
    alice.current_bet = 0
    to_call = game.current_bet - alice.current_bet
    with pytest.raises(ActionValidationError, match="Cannot fold when you can check"):
        game.handle_fold(alice, to_call)

def test_fold_when_not_in_hand_fails():
    game, alice, _ = setup_game()
    alice.in_hand = False
    game.current_bet = 100
    alice.current_bet = 50
    to_call = game.current_bet - alice.current_bet
    with pytest.raises(ActionValidationError, match="Cannot fold if player is not in hand."):
        game.handle_fold(alice, to_call)

def test_fold_with_negative_to_call_fails():
    game, alice, _ = setup_game()
    alice.in_hand = True
    game.current_bet = -10
    alice.current_bet = 0
    to_call = game.current_bet - alice.current_bet
    with pytest.raises(ActionValidationError):
        game.handle_fold(alice, to_call)

def test_fold_with_non_bool_in_hand_fails():
    game, alice, _ = setup_game()
    alice.in_hand = 1  # Should be bool
    game.current_bet = 100
    alice.current_bet = 50
    to_call = game.current_bet - alice.current_bet
    with pytest.raises(ActionValidationError):
        game.handle_fold(alice, to_call)

def test_fold_with_non_integer_to_call_fails():
    game, alice, _ = setup_game()
    alice.in_hand = True
    game.current_bet = "fifty"
    alice.current_bet = 0
    to_call = "fifty"  # Simulate bug
    with pytest.raises(ActionValidationError):
        game.handle_fold(alice, to_call)

# --- handle_check tests ---

def test_check_when_to_call_zero():
    game, alice, _ = setup_game()
    game.current_bet = 0
    alice.current_bet = 0
    to_call = game.current_bet - alice.current_bet
    result = game.handle_check(alice, to_call)
    assert result["can_check"]
    assert result["pot"] == 150  # pot unchanged
    assert result["current_bet"] == 50 or result["current_bet"] == 0  # depending on your setup

def test_check_when_to_call_positive_fails():
    game, alice, _ = setup_game()
    game.current_bet = 50
    alice.current_bet = 0
    to_call = game.current_bet - alice.current_bet
    # to_call = 50, cannot check
    with pytest.raises(ActionValidationError, match="Cannot check when there is a bet to call."):
        game.handle_check(alice, to_call)

def test_check_with_negative_to_call_fails():
    game, alice, _ = setup_game()
    game.current_bet = -10
    alice.current_bet = 0
    to_call = game.current_bet - alice.current_bet
    with pytest.raises(ActionValidationError):
        game.handle_check(alice, to_call)

def test_check_with_non_integer_to_call_fails():
    game, alice, _ = setup_game()
    # Simulate a bug: set current_bet to a string
    game.current_bet = "zero"
    alice.current_bet = 0
    to_call = "zero"
    with pytest.raises(ActionValidationError):
        game.handle_check(alice, to_call)

# --- handle_call tests ---

def test_call_with_enough_chips():
    game, alice, _ = setup_game()
    # Alice needs to call 50, has 1000
    to_call = game.current_bet - alice.current_bet
    result = game.handle_call(alice, to_call)
    assert alice.stack == 950
    assert alice.current_bet == 50
    assert result["call_amount"] == 50
    assert not result["is_all_in"]

def test_call_all_in_for_less():
    game, alice, _ = setup_game()
    alice.stack = 30
    game.current_bet = 80
    alice.current_bet = 50
    to_call = game.current_bet - alice.current_bet  # 30 to call, but only 30 in stack
    result = game.handle_call(alice, to_call)
    assert alice.stack == 0
    assert alice.current_bet == 80
    assert result["call_amount"] == 30
    assert result["is_all_in"]

def test_call_with_exact_stack():
    game, alice, _ = setup_game()
    alice.stack = 50
    # Alice needs to call 50, has exactly 50 (all-in, but not "for less")
    to_call = game.current_bet - alice.current_bet
    result = game.handle_call(alice, to_call)
    assert alice.stack == 0
    assert alice.current_bet == 50
    assert result["call_amount"] == 50
    assert result["is_all_in"]

def test_call_with_zero_stack_fails():
    game, alice, _ = setup_game()
    alice.stack = 0
    to_call = game.current_bet - alice.current_bet
    with pytest.raises(ActionValidationError, match="Player has no chips left to call."):
        game.handle_call(alice, to_call)

def test_call_with_negative_stack_fails():
    game, alice, _ = setup_game()
    alice.stack = -10
    to_call = game.current_bet - alice.current_bet
    with pytest.raises(ActionValidationError):
        game.handle_call(alice, to_call)

def test_call_with_negative_to_call_fails():
    game, alice, _ = setup_game()
    game.current_bet = -5
    to_call = game.current_bet - alice.current_bet
    with pytest.raises(ActionValidationError):
        game.handle_call(alice, to_call)

def test_call_when_to_call_zero_is_check_not_call():
    game, alice, _ = setup_game()
    game.current_bet = 0
    alice.current_bet = 0
    to_call = game.current_bet - alice.current_bet
    # Should not be able to "call" when to_call is zero; should be a check
    with pytest.raises(ActionValidationError):
        game.handle_call(alice, to_call)

# --- handle_raise tests ---

def test_valid_raise():
    game, alice, _ = setup_game()
    to_call = game.current_bet - alice.current_bet
    result = game.handle_raise(alice, raise_to=150, to_call=to_call)  # 50 to call + 100 raise
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
    to_call = game.current_bet - alice.current_bet
    result = game.handle_raise(alice, raise_to=alice.current_bet + alice.stack, to_call=to_call)
    assert alice.stack == 0
    assert alice.current_bet == 200
    assert result["is_all_in"]

def test_minimum_raise():
    game, alice, _ = setup_game()
    to_call = game.current_bet - alice.current_bet
    result = game.handle_raise(alice, raise_to=100, to_call=to_call)  # 50 to call + 50 min raise
    assert alice.stack == 900
    assert alice.current_bet == 100
    assert game.current_bet == 100
    assert game.last_raise_amount == 50
    assert not result["is_all_in"]

def test_raise_too_small_raises_error():
    game, alice, _ = setup_game()
    to_call = game.current_bet - alice.current_bet
    with pytest.raises(ActionValidationError):
        game.handle_raise(alice, raise_to=game.current_bet + 10, to_call=to_call)  # Too small

def test_raise_over_stack_raises_error():
    game, alice, _ = setup_game()
    to_call = game.current_bet - alice.current_bet
    with pytest.raises(ActionValidationError):
        game.handle_raise(alice, raise_to=alice.current_bet + alice.stack + 1, to_call=to_call)

def test_raise_equal_to_call_raises_error():
    game, alice, _ = setup_game()
    to_call = game.current_bet - alice.current_bet
    with pytest.raises(ActionValidationError):
        game.handle_raise(alice, raise_to=game.current_bet, to_call=to_call)

def test_player_marked_all_in_after_raise():
    game, alice, _ = setup_game()
    alice.stack = 100
    to_call = game.current_bet - alice.current_bet
    result = game.handle_raise(alice, raise_to=alice.current_bet + alice.stack, to_call=to_call)
    assert alice.all_in

def test_all_in_below_min_raise_allowed():
    game, alice, _ = setup_game()
    alice.stack = 60
    to_call = game.current_bet - alice.current_bet
    # All-in raise below min raise should be allowed if it's all-in
    result = game.handle_raise(alice, raise_to=alice.current_bet + alice.stack, to_call=to_call)
    assert result["is_all_in"]

def test_all_in_not_enough_to_call_raises_error():
    game, alice, _ = setup_game()
    alice.stack = 10
    game.current_bet = 100
    alice.current_bet = 50
    to_call = game.current_bet - alice.current_bet  # 50 to call, only 10 in stack
    with pytest.raises(ActionValidationError):
        game.handle_raise(alice, raise_to=alice.current_bet + alice.stack, to_call=to_call)

def test_handle_raise_returns_structured_result():
    game, alice, _ = setup_game()
    to_call = game.current_bet - alice.current_bet
    result = game.handle_raise(alice, raise_to=150, to_call=to_call)
    assert isinstance(result, dict)
    assert "raise_to" in result
    assert "is_all_in" in result