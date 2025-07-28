#!/usr/bin/env python3
"""
Detailed debug script to examine table and game state during stalling
"""

import os
import sys
import builtins
import time

def main():
    # Save original print
    original_print = builtins.print
    
    def debug_print(*args, **kwargs):
        text = ' '.join(str(arg) for arg in args)
        # Only show tournament flow and stall investigation messages
        if any(critical in text for critical in [
            'Step ', 'remaining', 'eliminated', 'STALL', 'Table', 'Player',
            'Current player', 'Active', 'Hand over', 'Game state',
            'ERROR:', 'Exception:', 'players_to_act', 'current_player_idx'
        ]):
            original_print(*args, **kwargs)
    
    try:
        from agents.sharky_agent import SharkyAgent
        from env.rule_based_tournament_env import create_rule_based_training_env
        
        # Apply targeted debugging
        builtins.print = debug_print
        
        # Create a smaller tournament for faster debugging
        env = create_rule_based_training_env(total_players=6)
        
        class SimpleTestAgent:
            def act(self, obs, action_mask=None, deterministic=True):
                if action_mask is None:
                    return 1  # Default to call/check
                if action_mask[1]:  # Can call/check
                    return 1
                elif action_mask[0]:  # Can fold
                    return 0
                else:
                    return 2  # Raise if only option
        
        test_agent = SimpleTestAgent()
        
        original_print("🔍 Starting detailed table state debug...")
        
        seed = int(time.time()) % (2**32 - 1)
        obs, info = env.reset(seed=seed)
        
        done = False
        steps = 0
        max_steps = 1000
        
        last_remaining = 0
        stall_counter = 0
        
        while not done and steps < max_steps:
            action_mask = info.get('action_mask', None)
            action = test_agent.act(obs, action_mask=action_mask)
            
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            remaining = len([p for p in env.unwrapped.all_players if p.stack > 0])
            eliminated = len(env.unwrapped.elimination_order)
            
            # Check for progress every 50 steps
            if steps % 50 == 0:
                original_print(f"Step {steps}: {remaining} remaining, {eliminated} eliminated, done={done}")
                
                # Detect stalling
                if remaining == last_remaining and remaining > 1:
                    stall_counter += 1
                    if stall_counter >= 10:  # 500 steps with no progress
                        original_print(f"🚨 DETAILED STALL ANALYSIS at step {steps}!")
                        original_print(f"Tournament state: {remaining} remaining, {eliminated} eliminated")
                        
                        # Get detailed table state
                        if env.unwrapped.active_table_id in env.unwrapped.tables:
                            table = env.unwrapped.tables[env.unwrapped.active_table_id]
                            game = table.game
                            
                            original_print(f"Table {table.table_id} details:")
                            original_print(f"  Total players: {len(table.players)}")
                            original_print(f"  Active count: {table.get_active_player_count()}")
                            original_print(f"  Table is_active: {table.is_active}")
                            original_print(f"  Game hand_over: {game.hand_over}")
                            original_print(f"  Game phase: {game.PHASES[game.phase_idx] if game.phase_idx < len(game.PHASES) else 'unknown'}")
                            original_print(f"  Current player idx: {game.current_player_idx}")
                            original_print(f"  Players to act: {len(game.players_to_act)}")
                            
                            # Show all players at table
                            for i, player in enumerate(table.players):
                                status = "CURRENT" if i == game.current_player_idx else ""
                                original_print(f"    Player {i}: {player.name}, stack: {player.stack}, in_hand: {player.in_hand}, all_in: {player.all_in} {status}")
                            
                            # Show players_to_act
                            if game.players_to_act:
                                original_print(f"  Players to act: {[p.name for p in game.players_to_act]}")
                            else:
                                original_print(f"  No players to act - this might be the issue!")
                                
                            # Check action mask
                            mask = info.get('action_mask', [])
                            original_print(f"  Action mask: {mask} (any legal: {any(mask)})")
                            
                        break
                else:
                    stall_counter = 0
                    
                last_remaining = remaining
            
            if truncated:
                original_print(f"Tournament truncated at step {steps}")
                break
        
        original_print(f"Debug finished: steps={steps}, done={done}")
        
    except Exception as e:
        builtins.print = original_print  # Restore for error messages
        original_print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        builtins.print = original_print

if __name__ == "__main__":
    main()
