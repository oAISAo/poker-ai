#!/usr/bin/env python3
"""
Test script to verify elimination messages don't appear as duplicates.
"""

from env.multi_table_tournament_env import MultiTableTournamentEnv
from agents.rule_based_agents import LoosePassiveAgent
import random

def test_elimination_messages():
    """Test that elimination messages only appear once per player."""
    
    # Set up a small tournament to force eliminations quickly
    num_players = 6
    
    # Initialize environment
    env = MultiTableTournamentEnv(
        total_players=num_players,
        max_players_per_table=9,
        hands_per_blind_level=2,  # Short levels to force eliminations
        starting_stack=100,  # Small stacks to force eliminations quickly
        blinds_schedule=[(5, 10, 0), (10, 20, 0), (20, 40, 0)]  # Aggressive blinds
    )
    
    print(f"Starting tournament with {num_players} players")
    print("Looking for elimination messages...")
    
    step_count = 0
    max_steps = 1000  # Limit steps to avoid infinite loops
    
    obs = env.reset()
    
    while not env._tournament_finished() and step_count < max_steps:
        step_count += 1
        
        # Take a random action
        action_mask = env.legal_action_mask()
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        
        if valid_actions:
            action = random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                break
        else:
            print("No valid actions available")
            break
    
    print(f"Tournament completed after {step_count} steps")
    
    # Check final standings
    elimination_order = env.elimination_order
    if elimination_order:
        print(f"Final elimination order ({len(elimination_order)} players eliminated):")
        for i, player in enumerate(elimination_order):
            position = len(elimination_order) - i  # Calculate position based on elimination order
            print(f"  {position}: {player.name}")
    
    print("Test completed - check above output for any duplicate elimination messages")

if __name__ == "__main__":
    test_elimination_messages()
