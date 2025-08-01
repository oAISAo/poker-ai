#!/usr/bin/env python3
"""
Comprehensive tests for betting round transitions and state consistency.
These tests target the exact state inconsistency patterns observed during Sharky training.

The issues we're reproducing:
1. TAG_1.current_bet (50) > game.current_bet (0) - Player bets persisting after bet reset
2. LAG_1.current_bet (80) > game.current_bet (60) - Game bet not tracking max player bet
3. Player_0.current_bet (1500) > game.current_bet (0) - Sharky agent bet state issues

Based on training output patterns, these occur during:
- After posting blinds
- After new hand setup  
- Start of tournament step actions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.game import PokerGame
from engine.player import Player
from env.multi_table_tournament_env import MultiTableTournamentEnv


def test_forced_inconsistency_detection_in_tournament():
    """
    Test Case 10: Deliberately introduce a state inconsistency after table balancing and assert detection
    """
    print("\n=== Test 10: Forced Inconsistency Detection in Tournament ===")
    env = MultiTableTournamentEnv(total_players=8, max_players_per_table=4)
    obs = env.reset()

    # Eliminate a player to trigger balancing
    table_ids = list(env.tables.keys())
    table0 = env.tables[table_ids[0]]
    eliminated_player = table0.players[0]
    eliminated_player.stack = 0
    eliminated_player.in_hand = False
    print(f"[DEBUG] Eliminated {eliminated_player.name} from table {table_ids[0]}")

    # Trigger table balancing
    env.balance_table(table_ids[0])
    print(f"[DEBUG] Table balancing triggered")

    # Deliberately introduce an inconsistency: set a player's bet higher than game.current_bet
    for tid, table in env.tables.items():
        game = table.game
        if table.players:
            table.players[0].current_bet = game.current_bet + 100
            print(f"[DEBUG] Forced inconsistency: {table.players[0].name}.current_bet = {table.players[0].current_bet}, game.current_bet = {game.current_bet}")
            # Assert that the inconsistency is detected
            is_consistent = game._validate_state_consistency(f"forced inconsistency on table {tid}")
            assert not is_consistent, f"Inconsistency should be detected on table {tid}" 

    print("✓ Forced inconsistency detection in tournament passed")

def test_blind_level_change_and_table_balancing():
    """
    Test Case 7: Simulate blind level increases, player elimination, and table balancing
    Ensures state consistency after each event
    """
    print("\n=== Test 7: Blind Level Change and Table Balancing ===")
    env = MultiTableTournamentEnv(total_players=8, max_players_per_table=4, blinds_schedule=[(10, 20, 0), (20, 40, 1), (40, 80, 1)])
    obs = env.reset()

    # Simulate blind level increases and player elimination
    for blind_level in range(len(env.blinds_schedule)):
        env.current_blind_level = blind_level
        env._increase_blinds_if_needed()
        print(f"[DEBUG] Blind level set to {env.blinds_schedule[blind_level]}")

        # Eliminate a player from the active table
        table = env.tables[env.active_table_id]
        if table.players:
            eliminated_player = table.players[0]
            eliminated_player.stack = 0
            eliminated_player.in_hand = False
            print(f"[DEBUG] Eliminated {eliminated_player.name}")

        # Trigger table balancing
        env.balance_table(env.active_table_id)
        print(f"[DEBUG] Table balancing triggered")

        # Check state consistency after each event
        for tid, table in env.tables.items():
            game = table.game
            assert game._validate_state_consistency(f"after blind level {blind_level}, elimination, and balancing on table {tid}")

    print("✓ Blind level change and table balancing consistency passed")

def test_player_elimination_and_rebuy():
    """
    Test Case 8: Simulate player busting out and re-entering (rebuy)
    Ensures stack and bet states are reset correctly
    """
    print("\n=== Test 8: Player Elimination and Rebuy ===")
    env = MultiTableTournamentEnv(total_players=6, max_players_per_table=6)
    obs = env.reset()

    table = env.tables[env.active_table_id]
    game = table.game
    eliminated_player = table.players[0]
    eliminated_player.stack = 0
    eliminated_player.in_hand = False
    eliminated_player.current_bet = 100
    print(f"[DEBUG] Eliminated {eliminated_player.name} with bet={eliminated_player.current_bet}")

    # Simulate rebuy
    eliminated_player.stack = 1000
    eliminated_player.in_hand = True
    eliminated_player.current_bet = 0
    print(f"[DEBUG] {eliminated_player.name} rebuys with stack={eliminated_player.stack}, bet={eliminated_player.current_bet}")

    # Reset hand and check state
    game.reset_for_new_hand(is_first_hand=False)
    print(f"[DEBUG] After hand reset with rebuy:")
    for p in table.players:
        print(f"  {p.name}: stack={p.stack}, current_bet={p.current_bet}, in_hand={p.in_hand}")
    assert game._validate_state_consistency("after rebuy and hand reset")
    print("✓ Player elimination and rebuy consistency passed")

def test_table_balancing_consistency():
    """
    Test Case 9: Simulate moving players between tables and verify no duplicate blind posting or bet inconsistencies
    """
    print("\n=== Test 9: Table Balancing Consistency ===")
    env = MultiTableTournamentEnv(total_players=8, max_players_per_table=4)
    obs = env.reset()

    # Eliminate a player to trigger balancing
    table_ids = list(env.tables.keys())
    table0 = env.tables[table_ids[0]]
    eliminated_player = table0.players[0]
    eliminated_player.stack = 0
    eliminated_player.in_hand = False
    print(f"[DEBUG] Eliminated {eliminated_player.name} from table {table_ids[0]}")

    # Trigger table balancing
    env.balance_table(table_ids[0])
    print(f"[DEBUG] Table balancing triggered")

    # Check for duplicate blind posting and bet inconsistencies
    for tid, table in env.tables.items():
        game = table.game
        posted_blinds = [p.current_bet for p in table.players if p.current_bet > 0]
        print(f"[DEBUG] Table {tid} posted blinds: {posted_blinds}")
        # No player should have duplicate blinds
        assert all(p.current_bet <= game.big_blind for p in table.players), f"Player posted duplicate or excessive blind on table {tid}"
        assert game._validate_state_consistency(f"after table balancing on table {tid}")

    print("✓ Table balancing consistency passed")

def test_blind_level_change_and_elimination():
    """
    Test Case 6: Simulate blind level increases and player elimination in tournament
    Ensures state consistency after blind changes and bust-outs
    """
    print("\n=== Test 6: Blind Level Change and Elimination ===")

    # Create a turbo tournament with increasing blinds
    env = MultiTableTournamentEnv(total_players=6, max_players_per_table=6, blinds_schedule=[(10, 20, 0), (20, 40, 1), (40, 80, 1)])
    obs = env.reset()

    # Simulate a few steps and force a blind level increase
    for blind_level in range(len(env.blinds_schedule)):
        env.current_blind_level = blind_level
        env._increase_blinds_if_needed()
        print(f"[DEBUG] Blind level set to {env.blinds_schedule[blind_level]}")

        # Simulate player elimination
        table = env.tables[env.active_table_id]
        game = table.game
        eliminated_player = table.players[0]
        eliminated_player.stack = 0
        eliminated_player.in_hand = False

        # Reset hand and check state
        game.reset_for_new_hand(is_first_hand=False)
        print(f"[DEBUG] After hand reset with elimination:")
        for p in table.players:
            print(f"  {p.name}: stack={p.stack}, current_bet={p.current_bet}, in_hand={p.in_hand}")

        assert game._validate_state_consistency(f"after blind level {blind_level} and elimination")

    print("✓ Blind level change and elimination consistency passed")

def test_blind_posting_state_consistency():
    """
    Test Case 1: Reproduce 'current_bet (50) > game.current_bet (0)' pattern
    This happens when blinds are posted but game.current_bet gets reset incorrectly
    """
    print("\n=== Test 1: Blind Posting State Consistency ===")
    
    # Create a 6-player game to match tournament conditions
    players = [Player(f"Player_{i}", stack=1000) for i in range(6)]
    game = PokerGame(players, small_blind=25, big_blind=50)
    
    # Set up for first hand (this automatically posts blinds)
    game.reset_for_new_hand(is_first_hand=True)
    
    print(f"[DEBUG] After reset_for_new_hand (blinds already posted):")
    print(f"  game.current_bet: {game.current_bet}")
    for i, p in enumerate(players):
        print(f"  {p.name}.current_bet: {p.current_bet}")
    
    # State should be consistent after blind posting
    assert game._validate_state_consistency("after blind posting test")
    
    # Check that BB posted correct amount
    bb_pos = (game.dealer_position + 2) % len(players)
    bb_player = players[bb_pos]
    assert bb_player.current_bet == 50, f"BB should have bet 50, has {bb_player.current_bet}"
    assert game.current_bet == 50, f"game.current_bet should be 50, is {game.current_bet}"
    
    print("✓ Blind posting state consistency passed")

def test_bet_reset_between_rounds():
    """
    Test Case 2: Reproduce bet reset issues between betting rounds
    This tests the _advance_phase() -> reset_bets() sequence
    """
    print("\n=== Test 2: Bet Reset Between Rounds ===")
    
    players = [Player(f"Player_{i}", stack=1000) for i in range(4)]
    game = PokerGame(players, small_blind=10, big_blind=20)
    
    # Set up preflop (this automatically posts blinds)
    game.reset_for_new_hand(is_first_hand=True)
    
    # Simulate some preflop action - player 0 raises to 60
    if game.current_player_idx is not None:
        current_player = players[game.current_player_idx]
        print(f"[DEBUG] Current player before action: {current_player.name}")
    else:
        print(f"[DEBUG] No current player set")
    
    # Manually set up a betting scenario
    players[0].current_bet = 60  # Player raises to 60
    players[1].current_bet = 20  # BB bet
    game.current_bet = 60
    
    print(f"[DEBUG] Before advance_phase (preflop -> flop):")
    print(f"  game.current_bet: {game.current_bet}")
    for p in players:
        print(f"  {p.name}.current_bet: {p.current_bet}")
    
    # Advance to flop - this should reset bets
    game._advance_phase()
    
    print(f"[DEBUG] After advance_phase (now flop):")
    print(f"  game.current_bet: {game.current_bet}")
    for p in players:
        print(f"  {p.name}.current_bet: {p.current_bet}")
    
    # After phase advance, all bets should be reset
    assert game.current_bet == 0, f"game.current_bet should be 0 after phase advance, is {game.current_bet}"
    for p in players:
        assert p.current_bet == 0, f"{p.name}.current_bet should be 0, is {p.current_bet}"
    
    # State should be consistent
    assert game._validate_state_consistency("after phase advance to flop")
    
    print("✓ Bet reset between rounds passed")

def test_multiple_betting_rounds_consistency():
    """
    Test Case 3: Test complete betting round cycle (preflop -> flop -> turn -> river)
    This ensures state remains consistent through all transitions
    """
    print("\n=== Test 3: Multiple Betting Rounds Consistency ===")
    
    players = [Player(f"Player_{i}", stack=2000) for i in range(3)]
    game = PokerGame(players, small_blind=10, big_blind=20)
    
    # Start hand (this automatically posts blinds)
    game.reset_for_new_hand(is_first_hand=True)
    
    phases_to_test = ["preflop", "flop", "turn", "river"]
    
    for phase_name in phases_to_test:
        current_phase = game.PHASES[game.phase_idx]
        print(f"[DEBUG] Testing phase: {current_phase}")
        
        # Validate current state
        assert game._validate_state_consistency(f"during {current_phase}")
        
        if current_phase != "river":  # Don't advance past river
            # Simulate some betting action in this phase
            if current_phase == "preflop":
                # BB is already posted, just check consistency
                pass
            else:
                # Post-flop rounds start with 0 bets
                assert game.current_bet == 0, f"Post-flop bet should be 0, is {game.current_bet}"
            
            # Advance to next phase
            game._advance_phase()
            
            # Deal community cards for visual verification
            if game.phase_idx == 1:  # flop
                game.deal_community_cards(3)
            elif game.phase_idx == 2:  # turn
                game.deal_community_cards(1)
            elif game.phase_idx == 3:  # river
                game.deal_community_cards(1)
    
    print("✓ Multiple betting rounds consistency passed")

def test_tournament_env_state_consistency():
    """
    Test Case 4: Test state consistency within MultiTableTournamentEnv
    This reproduces the tournament-specific state inconsistency patterns
    """
    print("\n=== Test 4: Tournament Environment State Consistency ===")
    
    # Create a small tournament to test
    env = MultiTableTournamentEnv(total_players=6, max_players_per_table=6)
    
    # Get initial observation
    obs = env.reset()
    
    # Verify active table has consistent state
    if env.active_table_id in env.tables:
        table = env.tables[env.active_table_id]
        game = table.game
        
        print(f"[DEBUG] Initial tournament state:")
        print(f"  Active table: {env.active_table_id}")
        print(f"  game.current_bet: {game.current_bet}")
        print(f"  Current player: {game.current_player_idx}")
        
        if game.current_player_idx is not None and game.current_player_idx < len(table.players):
            current_player = table.players[game.current_player_idx]
            print(f"  Current player bet: {current_player.current_bet}")
        
        # Validate initial state
        assert game._validate_state_consistency("initial tournament state")
    
    # Take a few actions and check state remains consistent
    for step in range(5):
        action_mask = env.legal_action_mask()
        if any(action_mask):
            # Take the first legal action
            action = next(i for i, legal in enumerate(action_mask) if legal)
            obs, reward, done, truncated, info = env.step(action)
            
            # Check state consistency after each step
            if env.active_table_id in env.tables:
                table = env.tables[env.active_table_id]
                game = table.game
                
                print(f"[DEBUG] After step {step + 1}:")
                print(f"  game.current_bet: {game.current_bet}")
                if game.current_player_idx is not None and game.current_player_idx < len(table.players):
                    current_player = table.players[game.current_player_idx]
                    print(f"  Current player bet: {current_player.current_bet}")
                
                # This should not trigger warnings
                is_consistent = game._validate_state_consistency(f"tournament step {step + 1}")
                assert is_consistent, f"State inconsistency detected at tournament step {step + 1}"
            
            if done:
                break
    
    print("✓ Tournament environment state consistency passed")

def test_artificial_inconsistency_detection():
    """
    Test Case 5: Verify our state validation correctly detects the specific patterns
    This confirms our validation logic catches the exact issues we see in training
    """
    print("\n=== Test 5: Artificial Inconsistency Detection ===")
    
    players = [Player(f"Player_{i}", stack=1000) for i in range(4)]
    game = PokerGame(players, small_blind=25, big_blind=50)
    
    # Set up initial state (this automatically posts blinds)
    game.reset_for_new_hand(is_first_hand=True)
    
    # Test Pattern 1: Player bet > game bet (like "TAG_1.current_bet (50) > game.current_bet (0)")
    print("[DEBUG] Testing Pattern 1: Player bet > game bet")
    game.current_bet = 0  # Artificially reset game bet
    players[1].current_bet = 50  # But leave player bet high
    
    is_consistent = game._validate_state_consistency("artificial pattern 1")
    assert not is_consistent, "Should detect player bet > game bet inconsistency"
    
    # Test Pattern 2: Game bet != max player bet (like "game.current_bet (60) != max player bet (80)")
    print("[DEBUG] Testing Pattern 2: Game bet != max player bet")
    players[1].current_bet = 80
    players[2].current_bet = 60  
    game.current_bet = 60  # Game bet is lower than max player bet
    
    is_consistent = game._validate_state_consistency("artificial pattern 2")
    assert not is_consistent, "Should detect game bet != max player bet inconsistency"
    
    # Fix the inconsistencies
    print("[DEBUG] Testing inconsistency fixing")
    fixed = game.fix_state_inconsistencies()
    assert fixed, "Should be able to fix inconsistencies"
    
    # Verify it's fixed
    is_consistent = game._validate_state_consistency("after fix")
    assert is_consistent, "State should be consistent after fix"
    
    print("✓ Artificial inconsistency detection passed")
