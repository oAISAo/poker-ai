#!/usr/bin/env python3
"""
Debug script to test rule-based agents thoroughly
"""

import sys
import os
import builtins

# Save original print function BEFORE any imports that might override it
original_print = builtins.print

# Add project root to Python path when running directly
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from env.rule_based_tournament_env import RuleBasedTournamentEnv
from agents.rule_based_agents import get_mixed_opponent_pool

# Restore print function after imports
builtins.print = original_print


def debug_rule_based_agents():
    """Debug rule-based agents in detail"""
    print("🔍 Debugging Rule-Based Agents...")
    
    # Create test environment
    env = RuleBasedTournamentEnv(total_players=6, max_players_per_table=6)
    
    # Create agents
    temp_env = RuleBasedTournamentEnv(total_players=2)
    agents = get_mixed_opponent_pool(temp_env, total_opponents=5)
    
    print(f"✅ Created {len(agents)} agents:")
    for agent in agents:
        print(f"  • {agent.name}: {agent.style}")
    
    # Test single hand scenario
    print("\n🃏 Testing Single Hand Scenario...")
    obs, info = env.reset()
    
    print(f"🎯 Environment state check:")
    print(f"  • Tables: {len(env.tables)}")
    print(f"  • Active table ID: {getattr(env, 'active_table_id', 'Not found')}")
    print(f"  • Players count: {len(env.all_players)}")
    
    if hasattr(env, 'tables') and len(env.tables) > 0:
        table = env.tables[0]
        game = table.game
        print(f"  • Current player idx: {game.current_player_idx}")
        print(f"  • Current bet: {game.current_bet}")
        print(f"  • Pot: {game.pot}")
        
        if game.current_player_idx < len(game.players):
            current_player = game.players[game.current_player_idx]
            print(f"  • Current player: {current_player.name}")
            print(f"  • Current player bet: {current_player.current_bet}")
            print(f"  • Hole cards: {getattr(current_player, 'hole_cards', 'None')}")
    
    # Test agent decisions step by step
    print("\n🤖 Testing Agent Decision Making...")
    action_mask = info.get('action_mask', [True, True, True])
    print(f"Action mask: {action_mask} (Fold, Call/Check, Raise)")
    
    # Test each agent
    for i, agent in enumerate(agents):
        try:
            print(f"\n--- Testing {agent.name} ({agent.style}) ---")
            
            # Set the agent's environment reference
            agent.env = env
            
            # Try to get hand strength (only for agents that have this method)
            if hasattr(env, 'tables') and len(env.tables) > 0:
                table = env.tables[0] 
                game = table.game
                if game.current_player_idx < len(game.players):
                    current_player = game.players[game.current_player_idx]
                    hole_cards = getattr(current_player, 'hole_cards', None)
                    print(f"  Hole cards for decision: {hole_cards}")
                    
                    if hole_cards and hasattr(agent, 'get_hand_strength'):
                        hand_strength = agent.get_hand_strength(hole_cards)
                        print(f"  Hand strength: {hand_strength:.3f}")
            
            # Get agent action
            action = agent.act(observation=obs, action_mask=action_mask)
            action_name = ['Fold', 'Call/Check', 'Raise'][action]
            print(f"  Decision: {action} ({action_name})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Debug completed!")


if __name__ == "__main__":
    debug_rule_based_agents()
