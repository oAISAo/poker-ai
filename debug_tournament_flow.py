#!/usr/bin/env python3
"""
Test to see if rule-based agents are actually being called during tournament play
"""

import sys
import os
import builtins

# Save original print function BEFORE any imports that might override it
original_print = builtins.print

# Add project root to Python path when running directly
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from env.rule_based_tournament_env import create_rule_based_training_env
from agents.sharky_agent import SharkyAgent

# Restore print function after imports
builtins.print = original_print


def test_tournament_flow():
    """Test a short tournament to see what's actually happening"""
    print("🔍 Testing Tournament Flow...")
    
    # Create environment
    env = create_rule_based_training_env(total_players=6)
    
    # Create Sharky agent
    sharky = SharkyAgent(env, version="test")
    
    # Initialize environment
    obs, info = env.reset()
    print(f"🎯 Tournament started with {len(env.all_players)} players")
    
    # Track actions for first few steps
    step_count = 0
    done = False
    
    while not done and step_count < 20:  # Limit to first 20 steps
        print(f"\n--- Step {step_count + 1} ---")
        
        # Get current game state
        if hasattr(env, 'tables') and len(env.tables) > 0:
            table = env.tables[0]
            game = table.game
            current_player = game.players[game.current_player_idx]
            
            print(f"Current player: {current_player.name}")
            print(f"Current bet: {game.current_bet}")
            print(f"Player's current bet: {current_player.current_bet}")
            print(f"Stack: {current_player.stack}")
            
            # Check if this player has a rule-based agent
            if hasattr(current_player, 'agent'):
                print(f"Has rule-based agent: {current_player.agent.style}")
        
        # Get action mask
        action_mask = info.get('action_mask', [True, True, True])
        print(f"Action mask: {action_mask}")
        
        # Get Sharky's action
        action = sharky.act(obs, action_mask=action_mask, deterministic=True)
        print(f"Sharky's action: {action} ({'Fold' if action == 0 else 'Call' if action == 1 else 'Raise'})")
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        
        step_count += 1
        
        if truncated:
            print("Tournament truncated")
            break
    
    print(f"\n✅ Tested {step_count} steps")


if __name__ == "__main__":
    test_tournament_flow()
