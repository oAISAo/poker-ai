import pytest
from engine.game import PokerGame
from engine.player import Player
from engine.action_validation import validate_raise, validate_call, ActionValidationError

def setup_game():
    alice = Player("Alice")
    bob = Player("Bob")
    game = PokerGame([alice, bob], small_blind=20, big_blind=40)
    alice.stack = 1000
    bob.stack = 1000
    # Post blinds: Alice is SB, Bob is BB
    game.collect_bet(alice, 20, suppress_log=True)
    game.collect_bet(bob, 40, suppress_log=True)
    game.current_player_idx = 0  # Alice acts first after blinds
    game.current_bet = 40
    game.last_raise_amount = 40
    return game, alice, bob

# --- 3-player BB ante setup and tests ---
def setup_game_3p_bb_ante():
    # Dealer = Alice (pos 0), SB = Bob (pos 1), BB = Carol (pos 2)
    alice = Player("Alice")
    bob = Player("Bob")
    carol = Player("Carol")
    game = PokerGame([alice, bob, carol], small_blind=30, big_blind=60, ante=1)
    alice.stack = 1000
    bob.stack = 1000
    carol.stack = 1000
    # Post BB ante: only Carol (BB) posts, amount = big blind
    game.collect_ante(carol, 60, suppress_log=True)
    # Post blinds
    game.collect_bet(bob, 30, suppress_log=True)   # SB
    game.collect_bet(carol, 60, suppress_log=True) # BB
    game.current_player_idx = 0  # Alice (UTG) acts first
    game.current_bet = 60
    game.last_raise_amount = 60
    return game, alice, bob, carol

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
    alice.in_hand = "yes"  # type: ignore # Should be bool, but Python allows it
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
    assert result["pot"] == 60  # pot unchanged
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
    # Alice needs to call 20, has 980
    to_call = game.current_bet - alice.current_bet
    result = game.handle_call(alice, to_call)
    assert alice.stack == 960
    assert alice.current_bet == 40
    assert result["call_amount"] == 20
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
    alice.stack = 20
    # Alice needs to call 50, has exactly 50 (all-in, but not "for less")
    to_call = game.current_bet - alice.current_bet
    result = game.handle_call(alice, to_call)
    assert alice.stack == 0
    assert alice.current_bet == 40
    assert result["call_amount"] == 20
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
    result = game.handle_raise(alice, raise_to=120, to_call=to_call)
    assert alice.stack == 880
    assert alice.current_bet == 120
    assert game.current_bet == 120
    assert game.last_raise_amount == 80
    assert game.pot == 160
    assert result["raise_to"] == 120
    assert not result["is_all_in"]

def test_raise_all_in():
    game, alice, _ = setup_game()
    alice.stack = 40
    to_call = game.current_bet - alice.current_bet
    result = game.handle_raise(alice, raise_to=alice.current_bet + alice.stack, to_call=to_call)
    assert alice.stack == 0
    assert alice.current_bet == 60
    assert result["is_all_in"]

def test_minimum_raise():
    game, alice, _ = setup_game()
    to_call = game.current_bet - alice.current_bet
    result = game.handle_raise(alice, raise_to=100, to_call=to_call)
    assert alice.stack == 900
    assert alice.current_bet == 100
    assert game.current_bet == 100
    assert game.last_raise_amount == 60
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

def test_bb_ante_posting():
    game, alice, bob, carol = setup_game_3p_bb_ante()
    # Only Carol (BB) should have posted ante
    assert alice.total_contributed == 0
    assert bob.total_contributed == 30
    assert carol.total_contributed == 60 + 60  # ante + BB
    assert game.pot == 30 + 60 + 60
    assert carol.stack == 1000 - 60 - 60

def test_3p_bb_ante_action_flow():
    game, alice, bob, carol = setup_game_3p_bb_ante()
    # Alice (UTG) calls 60
    to_call = game.current_bet - alice.current_bet
    result = game.handle_call(alice, to_call)
    assert alice.current_bet == 60
    # Bob (SB) calls 30
    to_call = game.current_bet - bob.current_bet
    result = game.handle_call(bob, to_call)
    assert bob.current_bet == 60
    # Carol (BB) checks
    to_call = game.current_bet - carol.current_bet
    result = game.handle_check(carol, to_call)
    assert result["can_check"]

def test_3p_bb_ante_raise_and_fold():
    game, alice, bob, carol = setup_game_3p_bb_ante()
    # Alice (UTG) raises to 180
    to_call = game.current_bet - alice.current_bet
    result = game.handle_raise(alice, raise_to=180, to_call=to_call)
    assert alice.current_bet == 180
    # Bob (SB) folds
    to_call = game.current_bet - bob.current_bet
    result = game.handle_fold(bob, to_call)
    assert not bob.in_hand
    # Carol (BB) calls 120
    to_call = game.current_bet - carol.current_bet
    result = game.handle_call(carol, to_call)
    assert carol.current_bet == 180

def test_3p_bb_ante_bb_stack_less_than_ante():
    # Carol (BB) has less than BB for ante
    alice = Player("Alice")
    bob = Player("Bob")
    carol = Player("Carol")
    game = PokerGame([alice, bob, carol], small_blind=30, big_blind=60, ante=1)
    alice.stack = 1000
    bob.stack = 1000
    carol.stack = 50  # Less than BB for ante
    # Post BB ante: Carol posts all-in (50)
    game.collect_ante(carol, 50, suppress_log=True)
    # Post blinds
    game.collect_bet(bob, 30, suppress_log=True)
    game.collect_bet(carol, 0, suppress_log=True)  # Can't post BB
    game.current_player_idx = 0
    game.current_bet = 0
    game.last_raise_amount = 60
    # Carol should be all-in after ante, not able to post BB
    assert carol.stack == 0
    assert carol.all_in
    assert carol.current_bet == 0
    assert carol.total_contributed == 50
    # Pot should be sum of all contributions
    assert game.pot == 30 + 50