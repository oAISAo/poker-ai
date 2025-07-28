"""
Test script for rule-based agents
"""

import sys
import os
import builtins

# Save original print function BEFORE any imports that might override it
original_print = builtins.print

# Add project root to Python path when running directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from env.multi_table_tournament_env import MultiTableTournamentEnv
from agents.rule_based_agents import get_mixed_opponent_pool

# Restore print function after imports
builtins.print = original_print


def test_rule_based_agents():
    """Test that rule-based agents work properly"""
    print("🧪 Testing Rule-Based Agents...")
    
    # Create test environment
    env = MultiTableTournamentEnv(total_players=6, max_players_per_table=6)
    
    # Create agents
    agents = get_mixed_opponent_pool(env, total_opponents=5)
    
    print(f"✅ Created {len(agents)} rule-based agents:")
    for agent in agents:
        print(f"  • {agent.name}: {agent.style}")
    
    # Test that agents can make decisions
    print("\n🎯 Testing agent decision making...")
    
    obs, info = env.reset()
    action_mask = info.get('action_mask', [True, True, True])
    
    for agent in agents:
        try:
            action = agent.act(observation=obs, action_mask=action_mask)
            print(f"  {agent.name}: Action {action} ({'Fold' if action == 0 else 'Call' if action == 1 else 'Raise'})")
        except Exception as e:
            print(f"  ❌ {agent.name}: Error - {e}")
    
    print("\n✅ Rule-based agents test completed!")


if __name__ == "__main__":
    test_rule_based_agents()
