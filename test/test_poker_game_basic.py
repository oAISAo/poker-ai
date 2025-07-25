# test/test_poker_game_basic.py
import pytest
from engine.game import PokerGame
from engine.player import Player
from utils.enums import GameMode
from engine.action_validation import ActionValidationError

def test_last_raise_amount_consistency_after_multiple_raises():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    carol = Player("Carol", stack=1000)

    game = PokerGame([alice, bob, carol], small_blind=10, big_blind=20)
    game.reset_for_new_hand()

    # Set action order: Alice -> Bob -> Carol

    # Alice raises to 50 (raise by 30)
    game.current_player_idx = 0
    game.step("raise", 50)
    assert game.current_bet == 50
    assert game.last_raise_amount == 30

    # Bob re-raises to 120 (raise by 70)
    game.current_player_idx = 1
    game.step("raise", 120)
    assert game.current_bet == 120
    assert game.last_raise_amount == 70

    # Carol re-raises to 300 (raise by 180)
    game.current_player_idx = 2
    game.step("raise", 300)
    assert game.current_bet == 300
    assert game.last_raise_amount == 180

    # Now Alice tries a re-raise — must be at least +180
    game.current_player_idx = 0
    with pytest.raises(ActionValidationError, match="Must raise by at least 180 chips"):
        game.step("raise", 450)

def test_raise_below_min_not_all_in_raises_error():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()

    # Bob is big blind, current_bet is 20. Alice tries to raise to 30 (only +10).
    game.current_player_idx = 0  # Alice
    with pytest.raises(ActionValidationError, match="Must raise by at least .* chips"):
        game.step("raise", 30)

def test_all_in_below_minimum_raise_is_allowed():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=35)  # BB, 15 left after posting BB
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.dealer_position = 0
    game.reset_for_new_hand()

    # Alice is SB, must call 10 to match Bob's BB
    game.current_player_idx = 0  # Alice's index (SB)
    to_call = game.current_bet - game.players[game.current_player_idx].current_bet
    assert to_call == 10
    game.step("call", 10)

    # Now it's Bob's turn (BB), he can check or go all-in (raise to 35)
    game.current_player_idx = 1  # Bob's index (BB)
    to_call = game.current_bet - game.players[game.current_player_idx].current_bet
    assert to_call == 0  # Both have current_bet == 20
    game.step("raise", 35)

    assert bob.stack == 0
    assert bob.all_in is True
    assert bob.current_bet == 35
    assert game.current_bet == 20  # All-in below min raise does not update current_bet
    assert game.last_raise_amount == 20  # Remains unchanged

def test_cannot_check_when_to_call():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=20)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.dealer_position = 0  # Alice is dealer/SB, Bob is BB
    game.reset_for_new_hand()

    # Alice is SB, should have to_call == 10, cannot check
    game.current_player_idx = 0  # Alice's index (SB)
    to_call = game.current_bet - game.players[game.current_player_idx].current_bet
    assert to_call == 10
    with pytest.raises(ActionValidationError, match="Cannot check when there is a bet to call."):
        game.step("check")

def test_call_exact_stack_triggers_all_in():
    alice = Player("Alice", stack=1000)  # SB
    bob = Player("Bob", stack=20)        # BB, all-in
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.dealer_position = 0  # Alice is dealer/SB, Bob is BB
    game.reset_for_new_hand()

    # Alice is SB, must call 10 to match Bob's all-in
    game.current_player_idx = 0  # Alice's index (SB)
    to_call = game.current_bet - game.players[game.current_player_idx].current_bet
    assert to_call == 10
    game.step("call", 10)

    # After Alice calls, both are all-in, hand should proceed to showdown and end
    assert game.hand_over is True or game.phase_idx == game.PHASES.index("showdown")
    assert game.pot == 40
    assert alice.stack == 1020  # 990 after call + 40 pot awarded
    assert bob.stack == 0


def test_check_when_not_allowed_raises_error():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()

    # Alice tries to check when BB is 20 and she hasn’t called
    game.current_player_idx = 0
    with pytest.raises(ActionValidationError, match="Cannot check when there is a bet to call."):
        game.step("check")

def test_betting_round_does_not_skip_big_blind():
    # 3 players: Dealer, SB, BB
    dealer = Player("Dealer", stack=1000)
    sb = Player("SmallBlind", stack=1000)
    bb = Player("BigBlind", stack=1000)
    game = PokerGame([dealer, sb, bb], small_blind=10, big_blind=20)
    game.reset_for_new_hand()

    # Dealer calls 20
    game.current_player_idx = 0
    game.step("call", 0)
    # SB calls 10 to match BB's 20
    game.current_player_idx = 1
    game.step("call", 0)
    # BB should now get a chance to act (raise/check)
    assert game.current_player_idx == 2
    assert bb.current_bet == 20
    # BB raises to 100
    game.step("raise", 100)
    assert bb.current_bet == 100
    assert bb.stack == 900
    # Now dealer and SB must both get a chance to act again
    assert game.players_to_act[0] == dealer
    assert game.players_to_act[1] == sb
    # Dealer folds
    game.current_player_idx = 0
    game.step("fold")
    # SB calls 80 more
    game.current_player_idx = 1
    game.step("call", 0)
    # Now betting round should be complete, move to next phase
    assert game.players_to_act == []
    assert game.phase_idx == 1  # flop

# def test_betting_round_only_ends_after_all_respond_to_raise():
#     # 4 players: A, B, C, D
#     a = Player("A", stack=1000)
#     b = Player("B", stack=1000)
#     c = Player("C", stack=1000)
#     d = Player("D", stack=1000)
#     game = PokerGame([a, b, c, d], small_blind=10, big_blind=20)
#     game.reset_for_new_hand()
#     # A calls 20
#     game.current_player_idx = 0
#     game.step("call", 0)
#     # B raises to 100
#     game.current_player_idx = 1
#     game.step("raise", 100)
#     # Now C, D, and A must all act before round ends
#     assert set(game.players_to_act) == {c, d, a}
#     # C folds
#     game.current_player_idx = 2
#     game.step("fold")
#     # D calls 100
#     game.current_player_idx = 3
#     game.step("call", 0)
#     # A folds
#     game.current_player_idx = 0
#     game.step("fold")
#     # Now only B and D remain, and betting round should be complete
#     assert game.players_to_act == []
#     assert game.phase_idx == 1  # flop

# def test_betting_round_resets_players_to_act_on_new_raise():
#     # 3 players: X, Y, Z
#     x = Player("X", stack=1000)
#     y = Player("Y", stack=1000)
#     z = Player("Z", stack=1000)
#     game = PokerGame([x, y, z], small_blind=10, big_blind=20)
#     game.reset_for_new_hand()
#     # X raises to 50
#     game.current_player_idx = 0
#     game.step("raise", 50)
#     # Y calls
#     game.current_player_idx = 1
#     game.step("call", 0)
#     # Z re-raises to 120
#     game.current_player_idx = 2
#     game.step("raise", 120)
#     # Now X and Y must both act again
#     assert set(game.players_to_act) == {x, y}
#     # X calls
#     game.current_player_idx = 0
#     game.step("call", 0)
#     # Y folds
#     game.current_player_idx = 1
#     game.step("fold")
#     # Now only Z and X remain, betting round should be complete
#     assert game.players_to_act == []
#     assert game.phase_idx == 1  # flop

# def test_betting_round_complete_when_all_in_players_are_skipped():
#     # 3 players: A, B, C
#     a = Player("A", stack=1000)
#     b = Player("B", stack=100)
#     c = Player("C", stack=1000)
#     game = PokerGame([a, b, c], small_blind=10, big_blind=20)
#     game.reset_for_new_hand()
#     # A raises to 200
#     game.current_player_idx = 0
#     game.step("raise", 200)
#     # B calls all-in (should be 80 more, total 100)
#     game.current_player_idx = 1
#     game.step("call", 0)
#     assert b.stack == 0 and b.all_in
#     # C calls 200
#     game.current_player_idx = 2
#     game.step("call", 0)
#     # Now only A, C remain, B is all-in and should be skipped
#     assert game.players_to_act == []
#     assert game.phase_idx == 1  # flop

# def test_players_to_act_resets_each_betting_round():
#     # 3 players: A, B, C
#     a = Player("A", stack=1000)
#     b = Player("B", stack=1000)
#     c = Player("C", stack=1000)
#     game = PokerGame([a, b, c], small_blind=10, big_blind=20)
#     game.reset_for_new_hand()
#     # A calls 20
#     game.current_player_idx = 0
#     game.step("call", 0)
#     # B calls 20
#     game.current_player_idx = 1
#     game.step("call", 0)
#     # C checks
#     game.current_player_idx = 2
#     game.step("check", 0)
#     # Betting round should be complete, players_to_act should be empty
#     assert game.players_to_act == []
#     assert game.phase_idx == 1  # flop
#     # Next round: A checks, B checks, C checks
#     game.current_player_idx = 0
#     game.step("check", 0)
#     game.current_player_idx = 1
#     game.step("check", 0)
#     game.current_player_idx = 2
#     game.step("check", 0)
#     assert game.players_to_act == []
#     assert game.phase_idx == 2  # turn

# def test_big_blind_raises_to_100():
#     dealer = Player("Dealer", stack=1000, is_human=False)
#     sb = Player("SmallBlind", stack=1000, is_human=False)
#     bb = Player("BigBlind", stack=1000, is_human=False)

#     game = PokerGame([dealer, sb, bb], small_blind=10, big_blind=20)
#     game.reset_for_new_hand()

#     # Dealer calls 20
#     game.current_player_idx = 0
#     game.step("call", 0)
#     assert dealer.current_bet == 20
#     assert dealer.stack == 980

#     # SB calls 10 to match BB's 20 (10 already posted)
#     game.current_player_idx = 1
#     game.step("call", 0)

#     # Instead of checking sb.current_bet, check pot and sb stack
#     assert sb.stack == 980
#     assert game.pot >= 30 + 10  # 10 SB call + 10 SB small blind + 20 BB

#     # BB raises to 100 total
#     game.current_player_idx = 2
#     game.step("raise", 100)

#     assert bb.current_bet == 100
#     assert bb.stack == 900
#     assert game.current_bet == 100
#     assert game.pot >= 100 + 20 + 20  # BB raise + dealer + sb calls

#     # Fold others so BB wins
#     game.current_player_idx = 0
#     game.step("fold")
#     game.current_player_idx = 1
#     game.step("fold")

#     assert game.hand_over is True
#     assert bb.stack == 900 + game.pot

#     print("Test passed: BB raise to 100 with correct bet and stack updates.")

# def test_raise_to_100_from_small_blind_position():
#     # Setup players: Dealer, SB, BB
#     dealer = Player("Dealer", stack=1000, is_human=False)
#     sb = Player("SmallBlind", stack=1000, is_human=False)
#     bb = Player("BigBlind", stack=1000, is_human=False)

#     game = PokerGame([dealer, sb, bb], small_blind=10, big_blind=20)
#     game.reset_for_new_hand()

#     # Post blinds are automatic on reset_for_new_hand:
#     # sb.current_bet == 10, bb.current_bet == 20, pot == 30

#     # Dealer calls 20 (to call BB)
#     game.current_player_idx = 0  # Dealer's turn
#     game.step("call", 0)
#     assert dealer.current_bet == 20
#     assert dealer.stack == 980

#     # SB raises to 100 total
#     game.current_player_idx = 1  # SB's turn
#     game.step("raise", 100)

#     assert sb.current_bet == 100
#     assert sb.stack == 900
#     assert game.current_bet == 100
#     assert game.pot >= 140  # 20 (BB) + 20 (Dealer) + 100 (SB raise)

#     # Let's fold the others so SB wins and collects the pot
#     game.current_player_idx = 2  # BB
#     game.step("fold")
#     game.current_player_idx = 0  # Dealer
#     game.step("fold")

#     # Hand should be over and SB should win pot
#     assert game.hand_over is True
#     assert sb.stack == 900 + game.pot  # 900 + 140

#     print("Test passed: SB raise to 100 with correct bet and stack updates.")

# def test_raise_to_100_sets_correct_current_bet():
#     # Setup two players
#     alice = Player("Alice", stack=1000, is_human=False)
#     bob = Player("Bob", stack=1000, is_human=False)

#     # Setup game
#     game = PokerGame([alice, bob], small_blind=10, big_blind=20)
#     game.reset_for_new_hand()

#     # It's Bob's turn, and current_bet is 20 (big blind)
#     game.current_player_idx = 1  # Bob's index

#     # Bob raises to 100 total
#     game.step("raise", 100)

#     # Assert Bob's current bet and stack updated
#     assert bob.current_bet == 100
#     assert bob.stack == 900
#     assert game.current_bet == 100
#     assert game.pot >= 100  # Pot updated accordingly

# def test_single_remaining_player_wins():
#     alice = Player("Alice", stack=1000)
#     bob = Player("Bob", stack=1000)
#     game = PokerGame([alice, bob])
#     game.reset_for_new_hand()

#     # Simulate one fold
#     current_idx = game.current_player_idx
#     folding_player = game.players[current_idx]
#     game.step("fold")

#     assert game.hand_over is True
#     assert folding_player.in_hand is False
#     winner = next(p for p in game.players if p.in_hand)
#     assert winner.stack > 1000  # won the pot


# def test_community_cards_dealing():
#     alice = Player("Alice", stack=1000)
#     bob = Player("Bob", stack=1000)
#     game = PokerGame([alice, bob])

#     game.reset_for_new_hand()

#     # Force progress to flop
#     game.phase_idx = 1  # Preflop -> Flop
#     game.deal_community_cards(3)
#     assert len(game.community_cards) == 3

#     # Turn
#     game.deal_community_cards(1)
#     assert len(game.community_cards) == 4

#     # River
#     game.deal_community_cards(1)
#     assert len(game.community_cards) == 5


# def test_dealer_rotation():
#     alice = Player("Alice", stack=1000)
#     bob = Player("Bob", stack=1000)
#     game = PokerGame([alice, bob])

#     initial_dealer = game.dealer_position
#     game.rotate_dealer()
#     assert game.dealer_position == (initial_dealer + 1) % 2


# def test_blinds_posting():
#     alice = Player("Alice", stack=1000)
#     bob = Player("Bob", stack=1000)
#     game = PokerGame([alice, bob], small_blind=10, big_blind=20)

#     game.reset_for_new_hand()

#     # Check blinds
#     sb_pos = (game.dealer_position + 1) % 2
#     bb_pos = (game.dealer_position + 2) % 2

#     sb_player = game.players[sb_pos]
#     bb_player = game.players[bb_pos]

#     assert sb_player.current_bet == 10
#     assert bb_player.current_bet == 20
#     assert game.pot == 30
#     assert sb_player.stack == 990
#     assert bb_player.stack == 980


def test_basic_game_flow():
    # Setup two players
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)

    # Create game instance (AI vs AI mode)
    game = PokerGame([alice, bob], game_mode=GameMode.AI_VS_AI)

    # Play a single hand
    game.play_hand()

    # Total stack must be conserved (ignoring rake or errors)
    total_stack_after = alice.stack + bob.stack
    assert total_stack_after == 2000

    # At least one player should have changed stack
    assert alice.stack != 1000 or bob.stack != 1000

    # Players must not have negative chips
    assert alice.stack >= 0
    assert bob.stack >= 0

    # Players' hole cards should be cleared
    assert alice.hole_cards == []
    assert bob.hole_cards == []

def test_player_fold_behavior():
    player = Player("TestPlayer", stack=1000)
    player.fold()
    assert player.in_hand is False

def test_player_bet_and_all_in():
    player = Player("TestPlayer", stack=50)
    bet = player.bet_chips(100)
    # Bet cannot exceed stack, so should be 50
    assert bet == 50
    assert player.all_in is True
    assert player.stack == 0

def test_player_reset_for_new_hand():
    player = Player("TestPlayer", stack=1000)
    player.bet_chips(100)
    player.fold()
    player.reset_for_new_hand()
    assert player.in_hand is True
    assert player.current_bet == 0
    assert player.all_in is False
    assert player.hole_cards == []

if __name__ == "__main__":
    pytest.main(["-v", __file__])
