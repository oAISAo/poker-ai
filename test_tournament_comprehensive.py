#!/usr/bin/env python3
"""
Comprehensive test to identify ActionValidationError patterns in tournaments.
"""

import sys
import random
from env.multi_table_tournament_env import MultiTableTournamentEnv

def test_tournament_until_error(max_steps=1000):
    """
    Run a tournament with mixed aggressive and passive agents until we hit an error
    or complete the tournament.
    """
    print("Starting comprehensive tournament test...")
    
    # Create a smaller tournament for faster testing
    env = MultiTableTournamentEnv(
        total_players=6,
        max_players_per_table=6,
        min_players_per_table=2,
        starting_stack=1000,
        hands_per_blind_level=5  # Faster blind increases
    )
    
    obs, info = env.reset(seed=42)
    
    # Mix of different action strategies to create more varied game states
    strategies = ["aggressive", "passive", "random", "tight", "loose", "mixed"]
    current_strategy = 0
    
    for step in range(max_steps):
        try:
            # Get legal actions
            action_mask = info.get("action_mask", [True, True, True])
            legal_actions = [i for i, legal in enumerate(action_mask) if legal]
            
            if not legal_actions:
                print(f"[WARNING] No legal actions at step {step}")
                action = 0  # Default to fold
            else:
                # Cycle through different strategies to create varied game states
                strategy = strategies[current_strategy % len(strategies)]
                
                if strategy == "aggressive":
                    # Prefer raising when possible, otherwise call
                    if len(legal_actions) >= 3 and 2 in legal_actions:
                        action = 2  # Raise
                    elif 1 in legal_actions:
                        action = 1  # Call
                    else:
                        action = legal_actions[0]
                elif strategy == "passive":
                    # Prefer calling/checking, avoid raising
                    if 1 in legal_actions:
                        action = 1  # Call/Check
                    else:
                        action = legal_actions[0]
                elif strategy == "tight":
                    # More likely to fold
                    if 0 in legal_actions and random.random() < 0.3:
                        action = 0  # Fold
                    elif 1 in legal_actions:
                        action = 1  # Call
                    else:
                        action = legal_actions[0]
                elif strategy == "loose":
                    # Less likely to fold, more likely to call/raise
                    if 2 in legal_actions and random.random() < 0.4:
                        action = 2  # Raise
                    elif 1 in legal_actions:
                        action = 1  # Call
                    else:
                        action = legal_actions[-1]  # Last legal action
                else:
                    # Random or mixed strategy
                    action = random.choice(legal_actions)
                
                # Change strategy every 10-20 steps to create variety
                if step % random.randint(10, 20) == 0:
                    current_strategy += 1
            
            # Execute the action
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 50 == 0:
                print(f"Step {step}: No errors so far, strategy: {strategy}")
                # Show tournament stats
                stats = env.get_tournament_stats()
                print(f"  Remaining players: {stats['remaining_players']}, Active tables: {stats['active_tables']}")
            
            if terminated:
                print(f"Tournament completed successfully at step {step}!")
                stats = env.get_tournament_stats()
                print(f"Final stats: {stats}")
                return "SUCCESS"
                
        except Exception as e:
            print(f"\n🚨 ERROR at step {step}: {type(e).__name__}: {e}")
            print(f"Strategy was: {strategy}")
            print(f"Action attempted: {action}")
            print(f"Legal actions: {legal_actions}")
            print(f"Action mask: {action_mask}")
            
            # Print current game state for debugging
            if hasattr(env, 'active_table_id') and env.active_table_id in env.tables:
                table = env.tables[env.active_table_id]
                if table.players and table.game.current_player_idx < len(table.players):
                    player = table.players[table.game.current_player_idx]
                    print(f"Current player: {player.name}")
                    print(f"Player stack: {player.stack}, current_bet: {player.current_bet}")
                    print(f"Game current_bet: {table.game.current_bet}, pot: {table.game.pot}")
                    print(f"Big blind: {table.game.big_blind}")
                    print(f"To call: {max(0, table.game.current_bet - player.current_bet)}")
            
            return f"ERROR: {type(e).__name__}: {e}"
    
    print(f"Test completed {max_steps} steps without errors (but tournament may not be finished)")
    stats = env.get_tournament_stats()
    print(f"Final stats: {stats}")
    return "TIMEOUT"

if __name__ == "__main__":
    result = test_tournament_until_error(max_steps=2000)
    print(f"\nTest result: {result}")
