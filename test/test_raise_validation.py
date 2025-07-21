import pytest
from engine.raise_validation import validate_raise

# Invalid: raise_to equals player's current bet (must raise)
# def test_raise_to_equal_player_current_bet_fails():
#     with pytest.raises(ValueError, match="must be greater than player's current bet"):
#         validate_raise(
#             raise_to=50,
#             player_stack=100,
#             to_call=50,
#             current_bet=50,
#             min_raise=20,
#             big_blind=20
#         )

# # Invalid: raise by less than min_raise increment
# def test_raise_by_less_than_min_increment_fails():
#     with pytest.raises(ValueError, match="must raise by at least 20"):
#         validate_raise(
#             raise_to=60,
#             player_stack=100,
#             to_call=50,
#             current_bet=0,
#             min_raise=20,
#             big_blind=20
#         )

# # Valid: raise by exactly the min_raise amount (minimum legal raise)
# def test_valid_raise_exact_min_increment():
#     validate_raise(
#         raise_to=70,
#         player_stack=100,
#         to_call=50,
#         current_bet=0,
#         min_raise=20,
#         big_blind=20
#     )

# # Invalid: raise exceeds stack
# def test_raise_exceeds_stack_fails():
#     with pytest.raises(ValueError, match="player only has 20 chips"):
#         validate_raise(
#             raise_to=80,
#             player_stack=20,
#             to_call=50,
#             current_bet=0,
#             min_raise=20,
#             big_blind=20
#         )

# Valid: all-in below min_raise allowed (as long as it's all-in)
def test_all_in_below_min_raise_allowed():
    validate_raise(
        raise_to=15,
        player_stack=5,
        to_call=5,          # Player needs 5 chips to call current bet of 15
        current_bet=15,     # Current highest bet on table is 15
        min_raise=20,
        big_blind=20,
        player_current_bet=10  # Player has already put 10 chips in pot
    )


# Invalid: player raises to amount they can't afford (not full all-in)
def test_invalid_partial_raise_not_all_in_fails():
    with pytest.raises(ValueError, match="player only has 15 chips"):
        validate_raise(
            raise_to=64,
            player_stack=15,
            to_call=50,
            current_bet=70,
            min_raise=20,
            big_blind=20,
            player_current_bet=20
        )


# 1. Raise less or equal player's current bet => fail
def test_raise_to_equal_or_less_than_current_bet_fails():
    with pytest.raises(ValueError, match="Raise must be greater than player's current bet"):
        validate_raise(raise_to=50, player_stack=100, to_call=20, current_bet=70, min_raise=20, big_blind=20, player_current_bet=50)

# 2. Raise amount exceeds player's stack => fail
def test_raise_exceeds_stack_fails():
    with pytest.raises(ValueError, match="player only has 10 chips"):
        validate_raise(raise_to=60, player_stack=10, to_call=20, current_bet=50, min_raise=20, big_blind=20, player_current_bet=30)

# 3. Raise smaller than min_raise (normal raise) => fail
def test_raise_smaller_than_min_raise_fails():
    with pytest.raises(ValueError, match="Must raise by at least 20 chips"):
        validate_raise(raise_to=65, player_stack=100, to_call=20, current_bet=50, min_raise=20, big_blind=20, player_current_bet=30)

# 4. Valid raise equal to min_raise => pass
def test_valid_raise_exact_min_raise_pass():
    validate_raise(raise_to=70, player_stack=100, to_call=20, current_bet=50, min_raise=20, big_blind=20, player_current_bet=30)

# 5. Opening bet less than big blind => fail unless all-in
def test_opening_bet_less_than_big_blind_fails():
    with pytest.raises(ValueError, match="Opening bet must be at least the big blind"):
        validate_raise(raise_to=10, player_stack=100, to_call=0, current_bet=0, min_raise=20, big_blind=20, player_current_bet=0)

# 6. Opening bet equal or greater than big blind => pass
def test_opening_bet_equal_to_big_blind_pass():
    validate_raise(raise_to=20, player_stack=100, to_call=0, current_bet=0, min_raise=20, big_blind=20, player_current_bet=0)

# 7. Opening bet less than big blind but all-in => pass
def test_opening_bet_less_than_big_blind_but_all_in_pass():
    validate_raise(raise_to=10, player_stack=10, to_call=0, current_bet=0, min_raise=20, big_blind=20, player_current_bet=0)

# 8. Raise with to_call=0 but raise_to less than current_bet => fail
def test_raise_to_less_than_current_bet_to_call_zero_fails():
    with pytest.raises(ValueError, match="Raise must be greater than player's current bet"):
        validate_raise(raise_to=90, player_stack=100, to_call=0, current_bet=100, min_raise=40, big_blind=20, player_current_bet=100)

# Test that a raise equal to or less than player's current bet fails
def test_raise_to_less_or_equal_current_bet_fails():
    """
    Player's current bet is effectively 80 (current_bet - to_call).
    A raise_to of 80 (not greater) should fail with "must be greater than player's current bet".
    """
    with pytest.raises(ValueError, match="Raise must be greater than player's current bet."):
        validate_raise(
            raise_to=80,
            player_stack=100,
            to_call=10,
            current_bet=90,
            min_raise=40,
            big_blind=20,
            player_current_bet=80
        )

