import pytest
from engine.action_validation import validate_raise, validate_call, validate_check, validate_fold, RaiseValidationError

# --- validate_check tests ---

def test_check_when_to_call_zero():
    """Player can check if to_call is zero."""
    result = validate_check(to_call=0)
    assert result["can_check"]

def test_check_when_to_call_positive_fails():
    """Player cannot check if to_call is greater than zero."""
    with pytest.raises(RaiseValidationError, match="Cannot check when there is a bet to call."):
        validate_check(to_call=10)

def test_check_with_negative_to_call_fails():
    """Negative to_call should fail."""
    with pytest.raises(RaiseValidationError):
        validate_check(to_call=-1)

def test_check_with_non_integer_to_call_fails():
    """Non-integer to_call should fail."""
    with pytest.raises(RaiseValidationError):
        validate_check(to_call="zero")

# --- validate_fold tests ---

def test_fold_when_in_hand_and_to_call_positive():
    """Player can fold if they are in hand and there is a bet to call."""
    result = validate_fold(in_hand=True, to_call=10)
    assert result["can_fold"]

def test_fold_when_in_hand_and_to_call_zero_fails():
    """Player cannot fold if they are in hand and to_call is zero (must check)."""
    with pytest.raises(RaiseValidationError, match="Cannot fold when you can check"):
        validate_fold(in_hand=True, to_call=0)

def test_fold_when_not_in_hand_fails():
    """Player cannot fold if they are not in hand."""
    with pytest.raises(RaiseValidationError, match="Cannot fold if player is not in hand."):
        validate_fold(in_hand=False, to_call=10)

def test_fold_with_negative_to_call_fails():
    """Negative to_call should fail."""
    with pytest.raises(RaiseValidationError):
        validate_fold(in_hand=True, to_call=-1)

def test_fold_with_non_bool_in_hand_fails():
    """Non-boolean in_hand should fail."""
    with pytest.raises(RaiseValidationError):
        validate_fold(in_hand=1, to_call=10)

def test_fold_with_non_integer_to_call_fails():
    """Non-integer to_call should fail."""
    with pytest.raises(RaiseValidationError):
        validate_fold(in_hand=True, to_call="ten")

# --- validate_call tests ---

def test_call_with_enough_chips():
    """Player can call the full amount if they have enough chips."""
    result = validate_call(player_stack=100, to_call=50)
    assert not result["is_all_in"]
    assert result["call_amount"] == 50

def test_call_all_in_for_less():
    """Player can call all-in for less if they don't have enough chips."""
    result = validate_call(player_stack=30, to_call=50)
    assert result["is_all_in"]
    assert result["call_amount"] == 30

def test_call_with_exact_stack():
    """Player can call exactly with their last chip (all-in)."""
    result = validate_call(player_stack=50, to_call=50)
    assert not result["is_all_in"]
    assert result["call_amount"] == 50

def test_call_with_zero_stack_fails():
    """Player with zero chips cannot call."""
    with pytest.raises(RaiseValidationError, match="Player has no chips left to call."):
        validate_call(player_stack=0, to_call=50)

def test_call_with_negative_stack_fails():
    """Negative stack should fail."""
    with pytest.raises(RaiseValidationError):
        validate_call(player_stack=-10, to_call=50)

def test_call_with_negative_to_call_fails():
    """Negative to_call should fail."""
    with pytest.raises(RaiseValidationError):
        validate_call(player_stack=100, to_call=-5)

# --- validate_raise tests ---

def test_raise_just_above_min_raise_pass():
    """Raise just above the minimum increment should pass."""
    result = validate_raise(
        raise_to=71,
        player_stack=100,
        to_call=20,
        current_bet=50,
        min_raise=20,
        big_blind=20,
        player_current_bet=30
    )
    assert not result.is_all_in

def test_raise_to_equal_player_current_bet_fails():
    """Raise to exactly player's current bet should fail."""
    with pytest.raises(RaiseValidationError, match="Raise must be greater than player's current bet."):
        validate_raise(
            raise_to=50,
            player_stack=100,
            to_call=50,
            current_bet=50,
            min_raise=20,
            big_blind=20,
            player_current_bet=50
        )

def test_valid_raise_exact_min_increment():
    """Raise by exactly the min_raise amount should pass."""
    result = validate_raise(
        raise_to=70,
        player_stack=100,
        to_call=50,
        current_bet=0,
        min_raise=20,
        big_blind=20,
        player_current_bet=0
    )
    assert not result.is_all_in

def test_raise_exact_stack_total_ok():
    """All-in with exactly enough to cover raise_to should pass."""
    result = validate_raise(
        raise_to=100,
        player_stack=20,
        to_call=20,  # <-- Fix: must be <= amount_to_put_in
        current_bet=80,
        min_raise=20,
        big_blind=20,
        player_current_bet=80
    )
    assert result.is_all_in

def test_all_in_below_min_raise_allowed():
    """All-in raise above current bet, but less than min_raise, should be allowed."""
    result = validate_raise(
        raise_to=16,
        player_stack=6,
        to_call=5,
        current_bet=15,
        min_raise=20,
        big_blind=20,
        player_current_bet=10
    )
    assert result.is_all_in

def test_invalid_partial_raise_not_all_in_fails():
    """Partial raise not all-in and not enough chips should fail."""
    with pytest.raises(RaiseValidationError, match="player only has 15 chips"):
        validate_raise(
            raise_to=64,
            player_stack=15,
            to_call=50,
            current_bet=70,
            min_raise=20,
            big_blind=20,
            player_current_bet=20
        )

def test_raise_to_equal_or_less_than_current_bet_fails():
    """Raise to less or equal player's current bet should fail."""
    with pytest.raises(RaiseValidationError, match="Raise must be greater than player's current bet."):
        validate_raise(raise_to=50, player_stack=100, to_call=20, current_bet=70, min_raise=20, big_blind=20, player_current_bet=50)

def test_raise_exceeds_stack_fails():
    """Raise amount exceeds player's stack should fail."""
    with pytest.raises(RaiseValidationError, match="player only has 10 chips"):
        validate_raise(raise_to=60, player_stack=10, to_call=20, current_bet=50, min_raise=20, big_blind=20, player_current_bet=30)

def test_raise_smaller_than_min_raise_fails():
    """Raise smaller than min_raise (normal raise) should fail."""
    with pytest.raises(RaiseValidationError, match="Must raise by at least 20 chips"):
        validate_raise(raise_to=65, player_stack=100, to_call=20, current_bet=50, min_raise=20, big_blind=20, player_current_bet=30)

def test_valid_raise_exact_min_raise_pass():
    """Raise equal to min_raise should pass."""
    result = validate_raise(raise_to=70, player_stack=100, to_call=20, current_bet=50, min_raise=20, big_blind=20, player_current_bet=30)
    assert not result.is_all_in

def test_opening_bet_less_than_big_blind_fails():
    """Opening bet less than big blind should fail unless all-in."""
    with pytest.raises(RaiseValidationError, match="Opening bet must be at least the big blind"):
        validate_raise(raise_to=10, player_stack=100, to_call=0, current_bet=0, min_raise=20, big_blind=20, player_current_bet=0)

def test_opening_bet_equal_to_big_blind_pass():
    """Opening bet equal to big blind should pass."""
    result = validate_raise(raise_to=20, player_stack=100, to_call=0, current_bet=0, min_raise=20, big_blind=20, player_current_bet=0)
    assert not result.is_all_in

def test_opening_bet_less_than_big_blind_but_all_in_pass():
    """Opening bet less than big blind but all-in should pass."""
    result = validate_raise(raise_to=10, player_stack=10, to_call=0, current_bet=0, min_raise=20, big_blind=20, player_current_bet=0)
    assert result.is_all_in

def test_raise_to_less_than_current_bet_to_call_zero_fails():
    """Raise with to_call=0 but raise_to less than current_bet should fail."""
    with pytest.raises(RaiseValidationError, match="Raise must be greater than player's current bet."):
        validate_raise(raise_to=90, player_stack=100, to_call=0, current_bet=100, min_raise=40, big_blind=20, player_current_bet=100)

def test_raise_to_less_or_equal_current_bet_fails():
    """Raise equal to or less than player's current bet should fail."""
    with pytest.raises(RaiseValidationError, match="Raise must be greater than player's current bet."):
        validate_raise(
            raise_to=80,
            player_stack=100,
            to_call=10,
            current_bet=90,
            min_raise=40,
            big_blind=20,
            player_current_bet=80
        )

def test_all_in_exactly_min_raise():
    """All-in that is exactly the minimum raise should pass."""
    result = validate_raise(
        raise_to=70,
        player_stack=40,
        to_call=30,
        current_bet=30,
        min_raise=40,
        big_blind=20,
        player_current_bet=30
    )
    assert result.is_all_in

def test_all_in_less_than_call_fails():
    """All-in that is less than a call should fail."""
    with pytest.raises(RaiseValidationError, match="All-in is not enough to call the current bet."):
        validate_raise(
            raise_to=15,
            player_stack=5,
            to_call=10,
            current_bet=20,
            min_raise=20,
            big_blind=20,
            player_current_bet=10
        )

def test_negative_raise_to_fails():
    """Negative raise_to should fail."""
    with pytest.raises(RaiseValidationError, match="raise_to must be positive."):
        validate_raise(
            raise_to=-10,
            player_stack=100,
            to_call=0,
            current_bet=0,
            min_raise=20,
            big_blind=20,
            player_current_bet=0
        )

def test_zero_raise_to_fails():
    """Zero raise_to should fail."""
    with pytest.raises(RaiseValidationError, match="raise_to must be positive."):
        validate_raise(
            raise_to=0,
            player_stack=100,
            to_call=0,
            current_bet=0,
            min_raise=20,
            big_blind=20,
            player_current_bet=0
        )

def test_player_current_bet_greater_than_current_bet_fails():
    """player_current_bet greater than current_bet should fail."""
    with pytest.raises(RaiseValidationError, match="player_current_bet cannot be greater than current_bet."):
        validate_raise(
            raise_to=100,
            player_stack=100,
            to_call=0,
            current_bet=50,
            min_raise=20,
            big_blind=20,
            player_current_bet=60
        )

def test_player_stack_zero_fails():
    """Raise with player_stack = 0 should fail."""
    with pytest.raises(RaiseValidationError, match="Player has no chips left to bet."):
        validate_raise(
            raise_to=10,
            player_stack=0,
            to_call=0,
            current_bet=0,
            min_raise=20,
            big_blind=20,
            player_current_bet=0
        )