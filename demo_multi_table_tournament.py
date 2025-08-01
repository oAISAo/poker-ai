#!/usr/bin/env python3
"""
Demo script for Multi-Table Tournament Environment

This script demonstrates the key features of the multi-table poker tournament:
- Multiple tables with automatic player distribution
- Table balancing as players are eliminated
- Blind level increases over time
- Tournament progression tracking

Run with: python demo_multi_table_tournament.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.multi_table_tournament_env import MultiTableTournamentEnv
import numpy as np

def run_tournament_demo(total_players=27, max_players_per_table=9, hands_per_blind_level=5):
    """Run a demonstration of the multi-table tournament"""
    
    print(f"=== Multi-Table Poker Tournament Demo ===")
    print(f"Players: {total_players}")
    print(f"Max per table: {max_players_per_table}")
    print(f"Hands per blind level: {hands_per_blind_level}")
    print()
    
    # Create tournament environment
    env = MultiTableTournamentEnv(
        total_players=total_players,
        max_players_per_table=max_players_per_table,
        hands_per_blind_level=hands_per_blind_level,
        table_balancing_threshold=6
    )
    
    # Reset and get initial state
    obs, info = env.reset()
    
    print("Initial Tournament State:")
    env.render()
    print("\nInitial Statistics:")
    stats = env.get_tournament_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Run tournament simulation
    step_count = 0
    hand_count = 0
    last_hand_count = 0
    
    while not env._tournament_finished() and step_count < 1000:
        # Get legal actions
        mask = info["action_mask"]
        legal_actions = [i for i, legal in enumerate(mask) if legal]
        
        if not legal_actions:
            print("No legal actions available - tournament might be finished")
            break
        
        # Simple strategy: random legal action
        action = np.random.choice(legal_actions)
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        # Check if hand completed (new hand started)
        current_hand_count = env.total_hands_played
        if current_hand_count > last_hand_count:
            hand_count += 1
            last_hand_count = current_hand_count
            
            # Print periodic updates
            if hand_count % 10 == 0:
                print(f"\n--- After {hand_count} hands ---")
                stats = env.get_tournament_stats()
                print(f"Players remaining: {stats['remaining_players']}")
                print(f"Active tables: {stats['active_tables']}")
                print(f"Blind level: {stats['current_blind_level']} ({stats['blinds']})")
                print(f"Chip leader: {stats['chip_leader']} ({stats['chip_leader_stack']} chips)")
        
        if terminated:
            print("\nTournament completed!")
            break
    
    # Final results
    print(f"\n=== Final Tournament Results ===")
    env.render()
    
    final_stats = env.get_tournament_stats()
    print("\nFinal Statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nTournament completed in {step_count} steps and {hand_count} hands")
    
    # Show elimination order (last few players)
    if env.elimination_order:
        print(f"\nLast 10 eliminations:")
        for i, player in enumerate(env.elimination_order[-10:]):
            place = len(env.elimination_order) - i
            print(f"  {place}: {player.name}")
    
    # Show final winner if tournament completed
    remaining_players = [p for p in env.all_players if p.stack > 0]
    if len(remaining_players) == 1:
        winner = remaining_players[0]
        print(f"\n🏆 WINNER: {winner.name} with {winner.stack} chips!")
    elif len(remaining_players) > 1:
        print(f"\nTournament incomplete - {len(remaining_players)} players remain")
    
    return env

def demonstrate_table_balancing():
    """Demonstrate table balancing functionality"""
    print("\n=== Table Balancing Demonstration ===")
    
    # Create tournament with uneven distribution
    env = MultiTableTournamentEnv(total_players=50, max_players_per_table=9)
    obs, info = env.reset()
    
    print("Initial table distribution:")
    for table_id, table in env.tables.items():
        if table.is_active:
            print(f"  Table {table_id}: {len(table.players)} players")
    
    # Simulate eliminations to trigger balancing
    print("\nSimulating eliminations to trigger table balancing...")
    
    # Eliminate players from table 0 to trigger balancing
    target_table = env.tables[0]
    players_to_eliminate = target_table.players[7:]  # Leave only 5 players
    
    for player in players_to_eliminate:
        player.stack = 0
    
    # Trigger balancing check
    env.balance_table()
    
    print("After table balancing:")
    for table_id, table in env.tables.items():
        if table.is_active:
            active_count = table.get_active_player_count()
            print(f"  Table {table_id}: {active_count} active players")

def demonstrate_blind_structure():
    """Demonstrate blind level progression"""
    print("\n=== Blind Structure Demonstration ===")
    
    env = MultiTableTournamentEnv(
        total_players=18,
        hands_per_blind_level=3  # Quick progression for demo
    )
    obs, info = env.reset()
    
    print("Blind progression:")
    for level, (sb, bb) in enumerate(env.blinds_schedule[:10]):  # Show first 10 levels
        print(f"  Level {level + 1}: {sb}/{bb}")
    
    print(f"\nStarting at level {env.current_blind_level + 1}: {env.blinds_schedule[env.current_blind_level]}")
    
    # Simulate hands to show blind increases
    for hand in range(12):  # Should trigger multiple blind increases
        # Force hand completion
        env.hands_played_this_level += 1
        env.total_hands_played += 1
        env._increase_blinds_if_needed()
        
        if hand % 3 == 2:  # Every 3 hands
            current_blinds = env.blinds_schedule[env.current_blind_level]
            print(f"  After {hand + 1} hands: Level {env.current_blind_level + 1} ({current_blinds})")

if __name__ == "__main__":
    # Run the main tournament demo
    env = run_tournament_demo(total_players=27, hands_per_blind_level=3)
    
    # Show additional demonstrations
    demonstrate_table_balancing()
    demonstrate_blind_structure()
    
    print("\n=== Demo Complete ===")
    print("The multi-table tournament environment supports:")
    print("✓ Multiple tables with automatic player distribution")
    print("✓ Table balancing when tables become too small")
    print("✓ Automatic blind level increases")
    print("✓ Tournament progression tracking")
    print("✓ Player elimination and placement tracking")
    print("✓ Realistic poker tournament structure")
