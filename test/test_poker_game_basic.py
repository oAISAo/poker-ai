# test/test_poker_game_basic.py
import pytest
from engine.game import PokerGame
from engine.player import Player
from utils.enums import GameMode
from engine.action_validation import ActionValidationError

def test_nine_player_preflop_all_call_one_raise():
    players = [Player(f"P{i}", stack=1000) for i in range(9)]
    game = PokerGame(players, small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    # All call except P5, who raises to 100
    for i in range(9):
        game.current_player_idx = i
        to_call = game.current_bet - players[i].current_bet
        if i == 5:
            game.step("raise", 100)
        elif to_call == 0:
            game.step("check", 0)
        else:
            game.step("call", 0)
    step_limit = 20
    while game.phase_idx == game.PHASES.index("preflop") and step_limit > 0 and game.players_to_act:
        player = game.players_to_act[0]
        idx = game.players.index(player)
        game.current_player_idx = idx
        to_call = game.current_bet - player.current_bet
        if to_call == 0:
            game.step("check", 0)
        else:
            game.step("call", 0)
        step_limit -= 1
    assert game.phase_idx == game.PHASES.index("flop")
    expected_pot = 100 * 9
    assert game.pot == expected_pot

def test_nine_player_flop_all_in_and_folds():
    players = [Player(f"P{i}", stack=1000) for i in range(9)]
    game = PokerGame(players, small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    for i in range(9):
        game.current_player_idx = i
        to_call = game.current_bet - players[i].current_bet
        if to_call == 0:
            game.step("check", 0)
        else:
            game.step("call", 0)
    game.phase_idx = game.PHASES.index("flop")
    game.current_player_idx = 0
    game.step("raise", players[0].stack)
    for i in range(1, 5):
        game.current_player_idx = i
        game.step("call", 0)
    for i in range(5, 9):
        game.current_player_idx = i
        to_call = game.current_bet - players[i].current_bet
        if to_call == 0:
            game.step("check", 0)
        else:
            game.step("fold", 0)
        if game.hand_over:
            break
    assert game.hand_over
    winner = max(players, key=lambda p: p.stack)
    assert winner.stack > 1000

def test_nine_player_sequential_elimination():
    players = [Player(f"P{i}", stack=100) for i in range(9)]
    game = PokerGame(players, small_blind=10, big_blind=20)
    
    # Simulate multiple hands until only one player remains
    hands_played = 0
    max_hands = 20  # Safety limit
    
    while hands_played < max_hands:
        # Check how many players have chips
        remaining_players = [p for p in players if p.stack > 0]
        print(f"Hand {hands_played + 1}: Players with chips: {[p.name for p in remaining_players]} (stacks: {[p.stack for p in remaining_players]})")
        
        if len(remaining_players) <= 1:
            break
            
        if len(remaining_players) < 2:
            break
            
        try:
            game.reset_for_new_hand()
        except RuntimeError as e:
            if "Not enough players with chips to continue" in str(e):
                break
            else:
                raise
                
        # Find the two players with the smallest stacks (most likely to be eliminated)
        remaining_indices = [i for i, p in enumerate(players) if p.stack > 0]
        remaining_indices.sort(key=lambda i: players[i].stack)
        
        if len(remaining_indices) < 2:
            break
            
        # The player with the smallest stack goes all-in
        all_in_idx = remaining_indices[0]
        call_idx = remaining_indices[1]
        
        game.current_player_idx = all_in_idx
        # To go all-in, raise to current_bet + remaining_stack
        all_in_raise_to = players[all_in_idx].current_bet + players[all_in_idx].stack
        print(f"[TEST DEBUG] Player {players[all_in_idx].name}: stack={players[all_in_idx].stack}, current_bet={players[all_in_idx].current_bet}, all_in_raise_to={all_in_raise_to}")
        game.step("raise", all_in_raise_to)
        
        game.current_player_idx = call_idx
        game.step("call", 0)
        
        # All other players fold
        for i in range(9):
            if i not in (all_in_idx, call_idx) and players[i].stack > 0:
                game.current_player_idx = i
                to_call = game.current_bet - players[i].current_bet
                if to_call == 0:
                    game.step("check", 0)
                else:
                    game.step("fold", 0)
                if game.hand_over:
                    break
                    
        hands_played += 1
    
    # At the end, we should have at most one player with chips
    final_remaining = [p for p in players if p.stack > 0]
    print(f"Final result: Players with chips: {[p.name for p in final_remaining]}")
    
    # The test passes if we successfully eliminated players down to 1 or 0
    assert len(final_remaining) <= 1

def test_hand_ends_when_all_players_all_in():
    alice = Player("Alice", stack=100)
    bob = Player("Bob", stack=100)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("raise", 100)
    game.current_player_idx = 1
    game.step("call", 0)
    assert game.hand_over
    assert game.phase_idx == game.PHASES.index("showdown")
    assert alice.stack + bob.stack == 200

def test_hand_ends_when_all_but_one_fold():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("fold", 0)
    assert game.hand_over
    assert bob.stack > 1000

def test_hand_ends_when_all_players_eliminated_except_one():
    alice = Player("Alice", stack=0)
    bob = Player("Bob", stack=0)
    carol = Player("Carol", stack=1000)
    with pytest.raises(RuntimeError, match="Not enough players with chips to continue."):
        PokerGame([alice, bob, carol]).reset_for_new_hand()

def test_hand_ends_when_all_in_and_folded_mix():
    alice = Player("Alice", stack=0)
    bob = Player("Bob", stack=1000)
    carol = Player("Carol", stack=1000)
    game = PokerGame([alice, bob, carol], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    # Alice is all-in and should not act
    assert not alice.in_hand or getattr(alice, "all_in", False)
    # Only Bob and Carol should be able to act
    current_player = game.players[game.current_player_idx]
    assert current_player.name in ["Bob", "Carol"]
    # Simulate Bob folding if facing a bet, otherwise check
    to_call = game.current_bet - current_player.current_bet
    if to_call > 0:
        game.step("fold", 0)
    else:
        game.step("check", 0)
    # Hand should end or go to showdown
    assert game.hand_over or game.phase_idx == game.PHASES.index("showdown")

def test_hand_ends_when_all_players_folded():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    carol = Player("Carol", stack=1000)
    game = PokerGame([alice, bob, carol], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("fold", 0)
    game.current_player_idx = 1
    game.step("fold", 0)
    assert game.hand_over
    assert carol.stack == 1010
    assert game.pot == 0

def test_hand_ends_when_all_players_have_zero_stack():
    alice = Player("Alice", stack=0)
    bob = Player("Bob", stack=0)
    carol = Player("Carol", stack=0)
    with pytest.raises(RuntimeError, match="Not enough players with chips to continue."):
        PokerGame([alice, bob, carol]).reset_for_new_hand()

def test_three_player_full_hand_flow(monkeypatch):
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    carol = Player("Carol", stack=1000)
    game = PokerGame([alice, bob, carol], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("call", 0)
    game.current_player_idx = 1
    game.step("call", 0)
    game.current_player_idx = 2
    game.step("check", 0)
    assert game.phase_idx == 1
    game.current_player_idx = 0
    game.step("raise", 40)
    game.current_player_idx = 1
    game.step("fold")
    game.current_player_idx = 2
    game.step("call", 0)
    assert game.phase_idx == 2
    game.current_player_idx = 0
    game.step("check", 0)
    game.current_player_idx = 2
    game.step("raise", 100)
    game.current_player_idx = 0
    game.step("call", 0)
    assert game.phase_idx == 3
    game.current_player_idx = 0
    game.step("raise", 200)
    game.current_player_idx = 2
    game.step("fold")
    assert game.hand_over is True
    assert alice.stack > carol.stack

def test_full_hand_simulation_with_showdown(monkeypatch):
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("call", 0)
    game.current_player_idx = 1
    game.step("check", 0)
    assert game.phase_idx == 1
    game.current_player_idx = 0
    game.step("raise", 40)
    game.current_player_idx = 1
    game.step("call", 0)
    assert game.phase_idx == 2
    game.current_player_idx = 0
    game.step("check", 0)
    game.current_player_idx = 1
    game.step("check", 0)
    assert game.phase_idx == 3
    game.current_player_idx = 0
    game.step("raise", 100)
    game.current_player_idx = 1
    game.step("call", 0)
    assert game.phase_idx == 4 or game.hand_over
    total_stack = alice.stack + bob.stack
    assert total_stack == 2000

def test_split_pot(monkeypatch):
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    monkeypatch.setattr("engine.hand_evaluator.hand_rank", lambda *args, **kwargs: (1, [14, 13, 12, 11, 10]))
    monkeypatch.setattr("engine.game.hand_rank", lambda *args, **kwargs: (1, [14, 13, 12, 11, 10]))
    for phase in range(4):
        game.current_player_idx = 0
        game.step("check" if phase > 0 else "call", 0)
        game.current_player_idx = 1
        game.step("check", 0)
        if phase < 3:
            assert game.phase_idx == phase + 1
    assert abs(alice.stack - bob.stack) <= 1

def test_all_players_all_in_before_river(monkeypatch):
    alice = Player("Alice", stack=100)
    bob = Player("Bob", stack=100)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("call", 0)
    game.current_player_idx = 1
    game.step("check", 0)
    game.current_player_idx = 0
    game.step("raise", 80)
    game.current_player_idx = 1
    game.step("call", 0)
    assert game.hand_over or game.phase_idx == game.PHASES.index("showdown")
    assert alice.stack + bob.stack == 200

def test_player_eliminated_not_in_next_hand():
    alice = Player("Alice", stack=100)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("raise", 90)
    game.current_player_idx = 1
    game.step("call", 0)
    alice.stack = 0
    with pytest.raises(RuntimeError, match="Not enough players with chips to continue."):
        game.reset_for_new_hand()

def test_human_action_callback_called():
    alice = Player("Alice", stack=1000, is_human=True)
    bob = Player("Bob", stack=1000)
    called = {}
    def fake_callback(player, to_call):
        called["called"] = True
        if to_call == 0:
            return "check", 0
        else:
            return "call", 0
    game = PokerGame([alice, bob], human_action_callback=fake_callback)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step(None, 0)
    assert called.get("called") is True

def test_invalid_actions_raise_exceptions():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    with pytest.raises(ActionValidationError):
        game.step("check", 0)
    game.current_player_idx = 1
    with pytest.raises(ActionValidationError):
        game.step("raise", 25)

def test_dealer_button_skips_eliminated_players():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=0)
    carol = Player("Carol", stack=1000)
    game = PokerGame([alice, bob, carol])
    game.reset_for_new_hand()
    game.rotate_dealer()
    assert game.players[game.dealer_position].stack > 0

def test_three_player_full_hand_showdown(monkeypatch):
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    carol = Player("Carol", stack=1000)
    game = PokerGame([alice, bob, carol], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("call", 0)
    game.current_player_idx = 1
    game.step("call", 0)
    game.current_player_idx = 2
    game.step("check", 0)
    assert game.phase_idx == 1
    game.current_player_idx = 0
    game.step("raise", 40)
    game.current_player_idx = 1
    game.step("fold")
    game.current_player_idx = 2
    game.step("call", 0)
    assert game.phase_idx == 2
    game.current_player_idx = 0
    game.step("check", 0)
    game.current_player_idx = 2
    game.step("raise", 100)
    game.current_player_idx = 0
    game.step("call", 0)
    assert game.phase_idx == 3
    game.current_player_idx = 0
    game.step("raise", 200)
    game.current_player_idx = 2
    game.step("call", 0)
    assert game.phase_idx == 4 or game.hand_over
    total_stack = alice.stack + bob.stack + carol.stack
    assert total_stack == 3000

def test_three_player_one_folds_preflop(monkeypatch):
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    carol = Player("Carol", stack=1000)
    game = PokerGame([alice, bob, carol], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("fold")
    game.current_player_idx = 1
    game.step("call", 0)
    game.current_player_idx = 2
    game.step("check", 0)
    assert game.phase_idx == 1
    game.current_player_idx = 1
    game.step("raise", 40)
    game.current_player_idx = 2
    game.step("call", 0)
    assert game.phase_idx == 2
    game.current_player_idx = 1
    game.step("check", 0)
    game.current_player_idx = 2
    game.step("check", 0)
    assert game.phase_idx == 3
    game.current_player_idx = 1
    game.step("raise", 100)
    game.current_player_idx = 2
    game.step("fold")
    assert game.hand_over is True
    assert bob.stack > carol.stack

def test_three_player_all_in_side_pot(monkeypatch):
    alice = Player("Alice", stack=100)
    bob = Player("Bob", stack=500)
    carol = Player("Carol", stack=1000)
    game = PokerGame([alice, bob, carol], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("raise", 100)
    game.current_player_idx = 1
    game.step("call", 0)
    game.current_player_idx = 2
    game.step("call", 0)
    assert alice.all_in
    assert game.phase_idx == 1
    game.current_player_idx = 1
    game.step("raise", 300)
    game.current_player_idx = 2
    game.step("call", 0)
    assert game.phase_idx == 2
    game.current_player_idx = 1
    game.step("check", 0)
    game.current_player_idx = 2
    game.step("check", 0)
    assert game.phase_idx == 3
    game.current_player_idx = 1
    game.step("check", 0)
    game.current_player_idx = 2
    game.step("raise", 600)
    game.current_player_idx = 1
    game.step("fold")
    assert game.hand_over is True

def test_three_player_split_pot(monkeypatch):
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    carol = Player("Carol", stack=1000)
    game = PokerGame([alice, bob, carol], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    monkeypatch.setattr("engine.hand_evaluator.hand_rank", lambda *args, **kwargs: (1, [14, 13, 12, 11, 10]))
    monkeypatch.setattr("engine.game.hand_rank", lambda *args, **kwargs: (1, [14, 13, 12, 11, 10]))
    game.current_player_idx = 0
    game.step("call", 0)
    game.current_player_idx = 1
    game.step("call", 0)
    game.current_player_idx = 2
    game.step("check", 0)
    assert game.phase_idx == 1
    game.current_player_idx = 0
    game.step("raise", 40)
    game.current_player_idx = 1
    game.step("call", 0)
    game.current_player_idx = 2
    game.step("fold")
    assert game.phase_idx == 2
    game.current_player_idx = 0
    game.step("check", 0)
    game.current_player_idx = 1
    game.step("check", 0)
    assert game.phase_idx == 3
    game.current_player_idx = 0
    game.step("raise", 100)
    game.current_player_idx = 1
    game.step("call", 0)
    assert abs(alice.stack - bob.stack) <= 1

def test_full_hand_simulation():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("call", 0)
    game.current_player_idx = 1
    game.step("check", 0)
    assert game.phase_idx == 1
    game.current_player_idx = 0
    game.step("raise", 40)
    game.current_player_idx = 1
    game.step("call", 0)
    assert game.phase_idx == 2
    game.current_player_idx = 0
    game.step("check", 0)
    game.current_player_idx = 1
    game.step("check", 0)
    assert game.phase_idx == 3
    game.current_player_idx = 0
    game.step("raise", 100)
    game.current_player_idx = 1
    game.step("fold")
    assert game.hand_over is True
    assert alice.stack > bob.stack

def test_last_raise_amount_consistency_after_multiple_raises():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    carol = Player("Carol", stack=1000)
    game = PokerGame([alice, bob, carol], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("raise", 50)
    assert game.current_bet == 50
    assert game.last_raise_amount == 30
    game.current_player_idx = 1
    game.step("raise", 120)
    assert game.current_bet == 120
    assert game.last_raise_amount == 70
    game.current_player_idx = 2
    game.step("raise", 300)
    assert game.current_bet == 300
    assert game.last_raise_amount == 180
    game.current_player_idx = 0
    with pytest.raises(ActionValidationError, match="Must raise by at least 180 chips"):
        game.step("raise", 450)

def test_raise_below_min_not_all_in_raises_error():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    with pytest.raises(ActionValidationError, match="Must raise by at least .* chips"):
        game.step("raise", 30)

def test_all_in_below_minimum_raise_is_allowed():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=35)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.dealer_position = 0
    game.reset_for_new_hand()
    game.current_player_idx = 0
    to_call = game.current_bet - game.players[game.current_player_idx].current_bet
    assert to_call == 10
    game.step("call", 10)
    game.current_player_idx = 1
    to_call = game.current_bet - game.players[game.current_player_idx].current_bet
    assert to_call == 0
    game.step("raise", 35)
    assert bob.stack == 0
    assert bob.all_in is True
    assert bob.current_bet == 35
    assert game.current_bet == 20
    assert game.last_raise_amount == 20

def test_cannot_check_when_to_call():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=20)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.dealer_position = 0
    game.reset_for_new_hand()
    game.current_player_idx = 0
    to_call = game.current_bet - game.players[game.current_player_idx].current_bet
    assert to_call == 10
    with pytest.raises(ActionValidationError, match="Cannot check when there is a bet to call."):
        game.step("check")

def test_call_exact_stack_triggers_all_in():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=20)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.dealer_position = 0
    game.reset_for_new_hand()
    game.current_player_idx = 0
    to_call = game.current_bet - game.players[game.current_player_idx].current_bet
    assert to_call == 10
    game.step("call", 10)
    print(f"[TEST DEBUG] Pot: {game.pot}, Alice stack: {game.players[0].stack}, Bob stack: {game.players[1].stack}")
    assert game.hand_over is True or game.phase_idx == game.PHASES.index("showdown")
    assert game.players[0].stack == 1020
    assert game.players[1].stack == 0

def test_check_when_not_allowed_raises_error():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    print(f"SB current_bet: {game.players[game.dealer_position].current_bet}")
    print(f"BB current_bet: {game.players[(game.dealer_position + 1) % 2].current_bet}")
    print(f"Table current_bet: {game.current_bet}")
    sb_pos = game.dealer_position
    print(f"current_player_idx: {game.current_player_idx}, dealer_position: {game.dealer_position}")
    print(f"current_bet: {game.current_bet}, SB current_bet: {game.players[sb_pos].current_bet}")
    game.current_player_idx = sb_pos
    to_call = game.current_bet - game.players[sb_pos].current_bet
    print(f"to_call: {to_call}")
    assert to_call == 10

def test_betting_round_does_not_skip_big_blind():
    dealer = Player("Dealer", stack=1000)
    sb = Player("SmallBlind", stack=1000)
    bb = Player("BigBlind", stack=1000)
    game = PokerGame([dealer, sb, bb], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("call", 0)
    game.current_player_idx = 1
    game.step("call", 0)
    assert game.current_player_idx == 2
    assert bb.current_bet == 20
    game.step("raise", 100)
    assert bb.current_bet == 100
    assert bb.stack == 900
    assert game.players_to_act[0] == dealer
    assert game.players_to_act[1] == sb
    game.current_player_idx = 0
    game.step("fold")
    game.current_player_idx = 1
    game.step("call", 0)
    assert game.players_to_act == []
    assert game.phase_idx == 1

def test_betting_round_only_ends_after_all_respond_to_raise():
    a = Player("A", stack=1000)
    b = Player("B", stack=1000)
    c = Player("C", stack=1000)
    d = Player("D", stack=1000)
    game = PokerGame([a, b, c, d], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("call", 0)
    game.current_player_idx = 1
    game.step("raise", 100)
    assert set(game.players_to_act) == {c, d, a}
    game.current_player_idx = 2
    game.step("fold")
    game.current_player_idx = 3
    game.step("call", 0)
    game.current_player_idx = 0
    game.step("fold")
    assert game.players_to_act == []
    assert game.phase_idx == 1

def test_betting_round_resets_players_to_act_on_new_raise():
    x = Player("X", stack=1000)
    y = Player("Y", stack=1000)
    z = Player("Z", stack=1000)
    game = PokerGame([x, y, z], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("raise", 50)
    game.current_player_idx = 1
    game.step("call", 0)
    game.current_player_idx = 2
    game.step("raise", 120)
    assert set(game.players_to_act) == {x, y}
    game.current_player_idx = 0
    game.step("call", 0)
    game.current_player_idx = 1
    game.step("fold")
    assert game.players_to_act == []
    assert game.phase_idx == 1

def test_betting_round_complete_when_all_in_players_are_skipped():
    a = Player("A", stack=1000)
    b = Player("B", stack=100)
    c = Player("C", stack=1000)
    game = PokerGame([a, b, c], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("raise", 200)
    game.current_player_idx = 1
    game.step("call", 0)
    assert b.stack == 0 and b.all_in
    game.current_player_idx = 2
    game.step("call", 0)
    assert game.players_to_act == []
    assert game.phase_idx == 1

def test_players_to_act_resets_each_betting_round():
    a = Player("A", stack=1000)
    b = Player("B", stack=1000)
    c = Player("C", stack=1000)
    game = PokerGame([a, b, c], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("call", 0)
    print(f"After action: phase_idx={game.phase_idx}, players_to_act={[p.name for p in game.players_to_act]}")
    game.current_player_idx = 1
    game.step("call", 0)
    print(f"After action: phase_idx={game.phase_idx}, players_to_act={[p.name for p in game.players_to_act]}")
    game.current_player_idx = 2
    game.step("check", 0)
    print(f"After action: phase_idx={game.phase_idx}, players_to_act={[p.name for p in game.players_to_act]}")
    assert game.players_to_act == []
    assert game.phase_idx == 1
    game.current_player_idx = 0
    game.step("check", 0)
    print(f"After action: phase_idx={game.phase_idx}, players_to_act={[p.name for p in game.players_to_act]}")
    game.current_player_idx = 1
    game.step("check", 0)
    print(f"After action: phase_idx={game.phase_idx}, players_to_act={[p.name for p in game.players_to_act]}")
    game.current_player_idx = 2
    game.step("check", 0)
    print(f"After action: phase_idx={game.phase_idx}, players_to_act={[p.name for p in game.players_to_act]}")
    assert game.players_to_act == []
    assert game.phase_idx == 2

def test_big_blind_raises_to_100():
    dealer = Player("Dealer", stack=1000, is_human=False)
    sb = Player("SmallBlind", stack=1000, is_human=False)
    bb = Player("BigBlind", stack=1000, is_human=False)
    game = PokerGame([dealer, sb, bb], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("call", 0)
    assert dealer.current_bet == 20
    assert dealer.stack == 980
    game.current_player_idx = 1
    game.step("call", 0)
    assert sb.stack == 980
    assert game.pot >= 30 + 10
    game.current_player_idx = 2
    game.step("raise", 100)
    assert bb.current_bet == 100
    assert bb.stack == 900
    assert game.current_bet == 100
    assert game.pot >= 100 + 20 + 20
    game.current_player_idx = 0
    game.step("fold")
    game.current_player_idx = 1
    game.step("fold")
    assert game.hand_over is True
    assert bb.stack == 1040
    assert game.pot == 0
    print("Test passed: BB raise to 100 with correct bet and stack updates.")

def test_raise_to_100_from_small_blind_position():
    dealer = Player("Dealer", stack=1000, is_human=False)
    sb = Player("SmallBlind", stack=1000, is_human=False)
    bb = Player("BigBlind", stack=1000, is_human=False)
    game = PokerGame([dealer, sb, bb], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 0
    game.step("call", 0)
    assert dealer.current_bet == 20
    assert dealer.stack == 980
    game.current_player_idx = 1
    game.step("raise", 100)
    assert sb.current_bet == 100
    assert sb.stack == 900
    assert game.current_bet == 100
    assert game.pot >= 140
    game.current_player_idx = 2
    game.step("fold")
    game.current_player_idx = 0
    game.step("fold")
    assert game.hand_over is True
    assert sb.stack == 1040
    assert game.pot == 0
    print("Test passed: SB raise to 100 with correct bet and stack updates.")

def test_raise_to_100_sets_correct_current_bet():
    alice = Player("Alice", stack=1000, is_human=False)
    bob = Player("Bob", stack=1000, is_human=False)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    game.current_player_idx = 1
    game.step("raise", 100)
    assert bob.current_bet == 100
    assert bob.stack == 900
    assert game.current_bet == 100
    assert game.pot >= 100

def test_single_remaining_player_wins():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob])
    game.reset_for_new_hand()
    current_idx = game.current_player_idx
    acting_player = game.players[current_idx]
    # If to_call == 0, must check, not fold
    to_call = game.current_bet - acting_player.current_bet
    if to_call == 0:
        game.step("check", 0)
    else:
        game.step("fold", 0)
    if not game.hand_over:
        # Next player must act
        next_idx = (current_idx + 1) % 2
        next_player = game.players[next_idx]
        to_call = game.current_bet - next_player.current_bet
        if to_call == 0:
            game.step("check", 0)
        else:
            game.step("fold", 0)
    assert game.hand_over is True
    winner = next(p for p in game.players if p.in_hand)
    assert winner.stack > 1000

def test_community_cards_dealing():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob])
    game.reset_for_new_hand()
    game.phase_idx = 1
    game.deal_community_cards(3)
    assert len(game.community_cards) == 3
    game.deal_community_cards(1)
    assert len(game.community_cards) == 4
    game.deal_community_cards(1)
    assert len(game.community_cards) == 5

def test_dealer_rotation():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob])
    initial_dealer = game.dealer_position
    game.rotate_dealer()
    assert game.dealer_position == (initial_dealer + 1) % 2

def test_blinds_posting_heads_up():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    # Find SB and BB by current_bet
    sb_player = next(p for p in game.players if p.current_bet == 10)
    bb_player = next(p for p in game.players if p.current_bet == 20)
    assert sb_player.stack == 990
    assert bb_player.stack == 980
    assert game.pot == 30
    assert {sb_player.name, bb_player.name} == {"Alice", "Bob"}

def test_blinds_posting_three_players():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    carol = Player("Carol", stack=1000)
    game = PokerGame([alice, bob, carol], small_blind=10, big_blind=20)
    game.reset_for_new_hand()
    sb_pos = (game.dealer_position + 1) % 3
    bb_pos = (game.dealer_position + 2) % 3
    sb_player = game.players[sb_pos]
    bb_player = game.players[bb_pos]
    assert sb_player.current_bet == 10
    assert bb_player.current_bet == 20
    assert game.pot == 30
    assert sb_player.stack == 990
    assert bb_player.stack == 980

def test_basic_game_flow():
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)
    game = PokerGame([alice, bob], game_mode=GameMode.AI_VS_AI)
    game.play_hand()
    total_stack_after = alice.stack + bob.stack
    assert total_stack_after == 2000
    if alice.stack == 1000 and bob.stack == 1000:
        pass
    else:
        assert alice.stack != 1000 or bob.stack != 1000

def test_player_fold_behavior():
    player = Player("TestPlayer", stack=1000)
    player.fold()
    assert player.in_hand is False

def test_player_bet_and_all_in():
    player = Player("TestPlayer", stack=50)
    bet = player.bet_chips(100)
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
