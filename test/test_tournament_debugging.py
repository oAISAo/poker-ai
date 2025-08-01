#!/usr/bin/env python3
"""
Targeted tests to debug tournament issues found in evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import builtins
# Preserve original print before imports that might override it
original_print = builtins.print

from env.rule_based_tournament_env import create_rule_based_training_env
from env.multi_table_tournament_env import MultiTableTournamentEnv
import numpy as np
import random

# Restore print functionality for our tests
builtins.print = original_print

def test_basic_tournament_progression():
    """Test if tournament actually progresses and eliminates players"""
    print("=== Testing Basic Tournament Progression ===")
    
    # Use smaller starting stacks and higher blinds for faster eliminations
    env = MultiTableTournamentEnv(total_players=6, max_players_per_table=6, 
                                  starting_stack=200,  # Much smaller stack
                                  blinds_schedule=[(10, 20, 0), (20, 40, 0), (50, 100, 1), (100, 200, 1)])  # More aggressive
    obs, info = env.reset(seed=12345)
    
    print(f"Initial: {len([p for p in env.all_players if p.stack > 0])} players with stacks")
    print(f"Initial stacks: {[(p.name, p.stack) for p in env.all_players]}")
    
    # Play many steps with more aggressive random actions
    for step in range(100):
        mask = info.get('action_mask', [False, False, False])
        if not any(mask):
            print(f"Step {step}: No legal actions available")
            break
            
        # Use random actions weighted towards betting/raising to create pressure
        legal_actions = [i for i, legal in enumerate(mask) if legal]
        if len(legal_actions) == 1:
            action = legal_actions[0]
        else:
            # Prefer call/raise over fold to create action
            weights = []
            for a in legal_actions:
                if a == 0:  # fold
                    weights.append(1)
                elif a == 1:  # call/check  
                    weights.append(3)
                elif a == 2:  # raise
                    weights.append(2)
            
            action = random.choices(legal_actions, weights=weights)[0]
        obs, reward, done, truncated, info = env.step(action)
        
        remaining = len([p for p in env.all_players if p.stack > 0])
        eliminated = len(env.elimination_order)
        
        if step % 20 == 0:
            print(f"Step {step}: {remaining} remaining, {eliminated} eliminated, done={done}, truncated={truncated}")
            
        if done or truncated:
            print(f"Tournament ended at step {step}: done={done}, truncated={truncated}")
            break
            
        if eliminated > 0:
            print(f"First elimination at step {step}!")
            break
    
    final_remaining = len([p for p in env.all_players if p.stack > 0])
    final_eliminated = len(env.elimination_order)
    print(f"Final: {final_remaining} remaining, {final_eliminated} eliminated")
    
    assert final_eliminated > 0, "At least one player should be eliminated in tournament progression"

def test_rule_based_tournament_progression():
    """Test if rule-based tournament progresses properly"""
    print("\n=== Testing Rule-Based Tournament Progression ===")
    
    # Use the same aggressive settings that worked in basic test
    env = create_rule_based_training_env(total_players=6, starting_stack=200,
                                       blinds_schedule=[(10, 20), (20, 40), (50, 100), (100, 200)])
    obs, info = env.reset(seed=12345)
    
    print(f"Players: {[p.name for p in env.all_players]}")
    print(f"Player_0 at index: {[p.name for p in env.all_players].index('Player_0')}")
    print(f"Initial stacks: {[(p.name, p.stack) for p in env.all_players]}")
    
    # Play many steps
    for step in range(100):
        mask = info.get('action_mask', [False, False, False])
        if not any(mask):
            print(f"Step {step}: No legal actions available")
            break
            
        action = next(i for i, legal in enumerate(mask) if legal)
        obs, reward, done, truncated, info = env.step(action)
        
        remaining = len([p for p in env.all_players if p.stack > 0])
        eliminated = len(env.elimination_order)
        
        if step % 20 == 0:
            print(f"Step {step}: {remaining} remaining, {eliminated} eliminated")
            current_table = env.tables.get(env.active_table_id)
            if current_table:
                idx = current_table.game.current_player_idx
                current_player = current_table.players[idx] if current_table.players and idx is not None and 0 <= idx < len(current_table.players) else None
                print(f"  Current player: {current_player.name if current_player else 'None'}")
            
        if done or truncated:
            print(f"Tournament ended at step {step}: done={done}, truncated={truncated}")
            break
            
        if eliminated > 0:
            print(f"First elimination at step {step}!")
            print(f"Eliminated players: {[p.name for p in env.elimination_order]}")
            break
    
    final_remaining = len([p for p in env.all_players if p.stack > 0])
    final_eliminated = len(env.elimination_order)
    print(f"Final: {final_remaining} remaining, {final_eliminated} eliminated")
    
    assert final_eliminated > 0, "At least one player should be eliminated in rule-based tournament"

def test_forced_elimination():
    """Test forced elimination to check tracking"""
    print("\n=== Testing Forced Elimination ===")
    
    env = create_rule_based_training_env(total_players=6, starting_stack=200,
                                       blinds_schedule=[(10, 20), (20, 40), (50, 100), (100, 200)])
    obs, info = env.reset(seed=12345)
    
    print(f"Before elimination: {len(env.elimination_order)} eliminated")
    
    # Force eliminate a player
    player_to_eliminate = env.all_players[1]  # Not Player_0
    print(f"Eliminating {player_to_eliminate.name} (stack: {player_to_eliminate.stack})")
    player_to_eliminate.stack = 0
    
    # Update elimination order
    env._update_elimination_order()
    
    print(f"After elimination: {len(env.elimination_order)} eliminated")
    print(f"Elimination order: {[p.name for p in env.elimination_order]}")
    
    # Check if tournament detection works
    print(f"Tournament finished: {env._tournament_finished()}")
    
    assert len(env.elimination_order) > 0, "Player should be added to elimination order"
    assert player_to_eliminate in env.elimination_order, "Eliminated player should be in elimination order"

def test_game_step_functionality():
    """Test if individual game steps work properly"""
    print("\n=== Testing Game Step Functionality ===")
    
    env = create_rule_based_training_env(total_players=4, starting_stack=200,
                                       blinds_schedule=[(10, 20), (20, 40), (50, 100), (100, 200)])
    obs, info = env.reset(seed=12345)
    
    # Get current table and game
    table = env.tables[env.active_table_id]
    game = table.game
    
    print(f"Current player index: {game.current_player_idx}")
    print(f"Total players at table: {len(table.players)}")
    idx = game.current_player_idx
    current_player_name = table.players[idx].name if idx is not None and 0 <= idx < len(table.players) else 'INVALID'
    print(f"Current player: {current_player_name}")
    print(f"Current bet: {game.current_bet}")
    print(f"Pot: {game.pot}")
    print(f"Hand over: {game.hand_over}")
    print(f"Phase index: {getattr(game, 'phase_idx', 'N/A')}")
    
    # Try a single step
    mask = info.get('action_mask', [False, False, False])
    print(f"Legal actions: {mask}")
    
    if any(mask):
        action = next(i for i, legal in enumerate(mask) if legal)
        print(f"Taking action: {['fold', 'call/check', 'raise'][action]}")
        
        # Save state before step
        before_stacks = {p.name: p.stack for p in env.all_players}
        
        obs, reward, done, truncated, info = env.step(action)
        
        # Check state after step
        after_stacks = {p.name: p.stack for p in env.all_players}
        stack_changes = {name: after_stacks[name] - before_stacks[name] for name in before_stacks}
        
        print(f"Stack changes: {stack_changes}")
        print(f"Reward: {reward}")
        idx = game.current_player_idx
        new_current_player = table.players[idx].name if idx is not None and 0 <= idx < len(table.players) else 'INVALID'
        print(f"New current player: {new_current_player}")
        print(f"Hand over: {game.hand_over}")
        
        # Validate that step completed successfully
        assert obs is not None, "Observation should not be None after step"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
    else:
        print("No legal actions available!")
        assert False, "Should have at least one legal action available at start of tournament"

def test_player_0_consistency():
    """Test that Player_0 stays at index 0"""
    print("\n=== Testing Player_0 Consistency ===")
    
    for trial in range(5):
        env = create_rule_based_training_env(total_players=8, starting_stack=200,
                                           blinds_schedule=[(10, 20), (20, 40), (50, 100), (100, 200)])
        obs, info = env.reset(seed=trial)
        
        player_names = [p.name for p in env.all_players]
        player_0_index = player_names.index('Player_0') if 'Player_0' in player_names else -1
        
        print(f"Trial {trial}: Player_0 at index {player_0_index}, first 3 players: {player_names[:3]}")
        
        if player_0_index != 0:
            print(f"ERROR: Player_0 not at index 0!")
            assert False, f"Player_0 should be at index 0, but found at {player_0_index}"
    
    print("Player_0 consistently at index 0 ✓")
    assert True  # Test passed

def run_all_tests():
    """Run all debugging tests"""
    print("🔍 Running Tournament Debugging Tests...")
    
    print("Running basic progression test...")
    test_basic_tournament_progression()
    print("Running rule-based progression test...")
    test_rule_based_tournament_progression()  
    print("Running forced elimination test...")
    test_forced_elimination()
    print("Running game step functionality test...")
    test_game_step_functionality()
    print("Running player 0 consistency test...")
    test_player_0_consistency()
    
    print("\n🎉 All tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()
