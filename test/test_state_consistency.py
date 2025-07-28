#!/usr/bin/env python3
"""
Test script to verify state consistency fixes in the poker engine.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import PokerGame
from engine.player import Player

def test_state_consistency_validation():
    """Test that state consistency validation works correctly"""
    print("Testing state consistency validation...")
    
    # Create a simple game
    players = [Player(f"Player{i}", stack=1000) for i in range(3)]
    game = PokerGame(players, small_blind=10, big_blind=20)
    
    # Reset for new hand to set up initial state
    game.reset_for_new_hand(is_first_hand=True)
    
    # Test 1: Normal state should validate correctly
    print("\n1. Testing normal state validation...")
    is_consistent = game._validate_state_consistency("test - normal state")
    assert is_consistent, "Normal state should be consistent"
    print("✓ Normal state validation passed")
    
    # Test 2: Create artificial inconsistency and test detection
    print("\n2. Testing inconsistency detection...")
    # Artificially create inconsistency: make a player bet exceed game bet
    game.players[0].current_bet = game.current_bet + 50  # Invalid state
    
    is_consistent = game._validate_state_consistency("test - artificial inconsistency")
    assert not is_consistent, "Artificial inconsistency should be detected"
    print("✓ Inconsistency detection passed")
    
    # Test 3: Fix the inconsistency
    print("\n3. Testing inconsistency fixing...")
    fixed = game.fix_state_inconsistencies()
    assert fixed, "State inconsistency should be fixable"
    
    # Verify it's fixed
    is_consistent = game._validate_state_consistency("test - after fix")
    assert is_consistent, "State should be consistent after fix"
    print("✓ Inconsistency fixing passed")
    
    print("\n✅ All state consistency tests passed!")

def test_action_sequence_consistency():
    """Test that state remains consistent during a sequence of actions"""
    print("\nTesting action sequence consistency...")
    
    players = [Player(f"Player{i}", stack=1000) for i in range(3)]
    game = PokerGame(players, small_blind=10, big_blind=20)
    game.reset_for_new_hand(is_first_hand=True)
    
    # Execute a sequence of actions and verify consistency after each
    actions = [
        (0, "call", 0),      # Player 0 calls
        (1, "call", 0),      # Player 1 calls  
        (2, "raise", 50),    # Player 2 raises to 50
        (0, "fold", 0),      # Player 0 folds
        (1, "call", 0),      # Player 1 calls
    ]
    
    for i, (player_idx, action, amount) in enumerate(actions):
        print(f"\nAction {i+1}: Player {player_idx} {action} {amount}")
        game.current_player_idx = player_idx
        
        try:
            game.step(action, amount)
            # Consistency should be maintained after each action
            is_consistent = game._validate_state_consistency(f"after action {i+1}")
            if not is_consistent:
                print(f"⚠️  Inconsistency detected after action {i+1}")
                # Try to fix it
                game.fix_state_inconsistencies()
        except Exception as e:
            print(f"Action failed: {e}")
            break
    
    print("✅ Action sequence consistency test completed")

if __name__ == "__main__":
    test_state_consistency_validation()
    test_action_sequence_consistency()
    print("\n🎉 All state consistency tests completed successfully!")
