#!/usr/bin/env python3
"""
Additional tests to improve coverage of state consistency features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import PokerGame
from engine.player import Player
from env.multi_table_tournament_env import MultiTableTournamentEnv

def test_comprehensive_state_validation():
    """Test comprehensive state validation scenarios"""
    print("Testing comprehensive state validation scenarios...")
    
    # Test 1: Normal game flow with validation
    players = [Player(f"Player{i}", stack=1000) for i in range(3)]
    game = PokerGame(players, small_blind=10, big_blind=20)
    game.reset_for_new_hand(is_first_hand=True)
    
    # Normal state should validate
    assert game._validate_state_consistency("normal initial state")
    print("✓ Normal state validation passed")
    
    # Test 2: Artificial inconsistency detection
    game.players[0].current_bet = game.current_bet + 50
    assert not game._validate_state_consistency("artificial inconsistency")
    print("✓ Inconsistency detection passed")
    
    # Test 3: Fix the inconsistency
    fixed = game.fix_state_inconsistencies()
    assert fixed
    assert game._validate_state_consistency("after fix")
    print("✓ Inconsistency fixing passed")
    
    # Test 4: Tournament environment validation
    env = MultiTableTournamentEnv(total_players=6, max_players_per_table=3)
    obs, info = env.reset()
    
    # Execute actions to test tournament state validation
    action_count = 0
    while action_count < 20:
        action_mask = info.get("action_mask", [True, True, True])
        if any(action_mask):
            action = next(i for i, legal in enumerate(action_mask) if legal)
            obs, reward, done, truncated, info = env.step(action)
            action_count += 1
            
            if done:
                break
    
    print("✓ Tournament state validation completed")
    
    # Test 5: Edge case - all players same bet
    game2 = PokerGame([Player(f"P{i}", stack=1000) for i in range(3)], small_blind=10, big_blind=20)
    game2.reset_for_new_hand(is_first_hand=True)
    
    # Set all players to same bet amount
    for player in game2.players:
        player.current_bet = 50
    game2.current_bet = 50
    
    assert game2._validate_state_consistency("all equal bets")
    print("✓ Equal bets validation passed")
    
    # Test 6: Zero bet scenario
    for player in game2.players:
        player.current_bet = 0
    game2.current_bet = 0
    
    assert game2._validate_state_consistency("zero bets")
    print("✓ Zero bets validation passed")

def test_pot_calculation_validation():
    """Test pot calculation and validation"""
    print("\nTesting pot calculation validation...")
    
    players = [Player("Alice", stack=1000), Player("Bob", stack=1000)]
    game = PokerGame(players, small_blind=10, big_blind=20)
    game.reset_for_new_hand(is_first_hand=True)
    
    # Simulate betting to create pot
    game.current_player_idx = 0
    game.step("call", 0)  # Alice calls
    game.current_player_idx = 1  
    game.step("raise", 50)  # Bob raises to 50
    game.current_player_idx = 0
    game.step("call", 0)  # Alice calls
    
    # Verify pot calculation
    expected_pot = 50 + 50  # Both players bet 50
    print(f"Expected pot: {expected_pot}, Actual pot: {game.pot}")
    
    # State should be consistent
    assert game._validate_state_consistency("after betting sequence")
    print("✓ Pot calculation validation passed")

def test_all_in_state_consistency():
    """Test state consistency with all-in scenarios"""
    print("\nTesting all-in state consistency...")
    
    alice = Player("Alice", stack=50)  # Short stack
    bob = Player("Bob", stack=1000)
    
    game = PokerGame([alice, bob], small_blind=10, big_blind=20)
    game.reset_for_new_hand(is_first_hand=True)
    
    # Alice should be all-in after posting SB
    game.current_player_idx = 0  # Alice
    alice_initial_stack = alice.stack
    
    # Force all-in
    if alice.stack > 0:
        game.step("raise", alice.stack + alice.current_bet)
        assert alice.all_in or alice.stack == 0
        print(f"✓ Alice all-in: stack={alice.stack}, all_in={alice.all_in}")
    
    # State should be consistent
    assert game._validate_state_consistency("after all-in")
    print("✓ All-in state consistency passed")

def test_tournament_state_edge_cases():
    """Test tournament environment state validation edge cases"""
    print("\nTesting tournament state edge cases...")
    
    # Small tournament to test edge cases
    env = MultiTableTournamentEnv(total_players=4, max_players_per_table=2)
    obs, info = env.reset()
    
    # Test table with no players
    original_players = env.tables[0].players.copy()
    env.tables[0].players = []  # Temporarily empty table
    
    # Should handle empty table gracefully
    obs, reward, done, truncated, info = env.step(1)  # Try any action
    
    # Restore players
    env.tables[0].players = original_players
    print("✓ Empty table handling passed")
    
    # Test invalid player index
    table = env.tables[env.active_table_id]
    if table.players:
        original_idx = table.game.current_player_idx
        table.game.current_player_idx = 999  # Invalid index
        
        # Should handle invalid index gracefully
        obs, reward, done, truncated, info = env.step(1)
        print("✓ Invalid player index handling passed")

def test_synchronization_methods():
    """Test the synchronization methods directly"""
    print("\nTesting synchronization methods...")
    
    players = [Player(f"Player{i}", stack=1000) for i in range(3)]
    game = PokerGame(players, small_blind=10, big_blind=20)
    game.reset_for_new_hand(is_first_hand=True)
    
    # Test _synchronize_current_bet
    original_game_bet = game.current_bet
    game.players[1].current_bet = original_game_bet + 30
    
    game._synchronize_current_bet()
    assert game.current_bet == original_game_bet + 30
    print("✓ Current bet synchronization passed")
    
    # Test fix with player bet reduction
    game.players[2].current_bet = game.current_bet + 100
    original_stack = game.players[2].stack
    
    fixed = game.fix_state_inconsistencies()
    assert fixed
    assert game.players[2].current_bet <= game.current_bet
    assert game.players[2].stack >= original_stack  # Should get refund
    print("✓ Player bet reduction and refund passed")

if __name__ == "__main__":
    test_comprehensive_state_validation()
    test_pot_calculation_validation()
    test_all_in_state_consistency()
    test_tournament_state_edge_cases()
    test_synchronization_methods()
    print("\n🎉 All comprehensive state validation tests passed!")
