import pytest
from engine.raise_validation import validate_raise

def test_raise_to_less_than_or_equal_current_bet():
    with pytest.raises(ValueError, match="must be greater than player's current bet"):
        validate_raise(current_bet=50, last_raise_amount=20, player_current_bet=50, raise_to=50, player_stack=100)

def test_raise_by_less_than_last_raise_amount():
    with pytest.raises(ValueError, match="must raise by at least 20"):
        validate_raise(current_bet=50, last_raise_amount=20, player_current_bet=0, raise_to=60, player_stack=100)

def test_valid_raise_equal_to_last_raise_amount():
    # Should pass, raise exactly min raise size
    validate_raise(current_bet=50, last_raise_amount=20, player_current_bet=0, raise_to=70, player_stack=100)

def test_raise_but_not_enough_chips():
    with pytest.raises(ValueError, match="player only has 20 chips"):
        validate_raise(current_bet=50, last_raise_amount=20, player_current_bet=0, raise_to=80, player_stack=20)

def test_all_in_below_min_raise_allowed():
    # Player only has 15 chips left, min raise 20, but can go all-in with 15
    # This should pass (allowed)
    validate_raise(current_bet=50, last_raise_amount=20, player_current_bet=0, raise_to=15, player_stack=15)

def test_all_in_not_exact_stack_fails():
    # Player tries to raise to 64 but only has 15; not enough for min raise, and not a full all-in
    with pytest.raises(ValueError, match="player only has 15 chips"):
        validate_raise(current_bet=50, last_raise_amount=20, player_current_bet=0, raise_to=64, player_stack=15)

