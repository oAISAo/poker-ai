#!/usr/bin/env python3
"""
Debug script to investigate tournament stalling issue
"""

import os
import sys
import builtins
import time

def main():
    # Save original print and create a more targeted filter
    original_print = builtins.print
    
    def debug_print(*args, **kwargs):
        text = ' '.join(str(arg) for arg in args)
        # Only show critical tournament flow messages
        if any(critical in text for critical in [
            'Tournament hit step limit',
            'Step ', 'remaining', 'eliminated',
            'Player ', 'finished', 'place',
            'ERROR:', 'Exception:', 'Traceback',
            'table balancing', 'Table ', 'broken',
            'Active Tables:', 'Players remaining:',
            'Current player idx:', 'Invalid player', 'No legal actions'
        ]):
            original_print(*args, **kwargs)
        # Suppress everything else to focus on flow
    
    try:
        from agents.sharky_agent import SharkyAgent
        from env.rule_based_tournament_env import create_rule_based_training_env
        
        # Apply targeted debugging
        builtins.print = debug_print
        
        # Create a smaller tournament for faster debugging
        env = create_rule_based_training_env(total_players=6)  # Much smaller
        
        # Create a more aggressive test agent for faster tournaments
        class SimpleTestAgent:
            def act(self, obs, action_mask=None, deterministic=True):
                import random
                if action_mask is None:
                    return 1  # Default to call/check
                
                # Be more aggressive to speed up tournaments
                if action_mask[2]:  # Can raise
                    if random.random() < 0.3:  # 30% chance to raise
                        return 2
                
                if action_mask[1]:  # Can call/check
                    return 1
                elif action_mask[0]:  # Can fold
                    return 0
                else:
                    return 2  # Raise if only option
        
        test_agent = SimpleTestAgent()
        
        original_print("🔍 Starting small tournament debug...")
        original_print(f"Tournament size: 6 players")
        
        seed = int(time.time()) % (2**32 - 1)
        obs, info = env.reset(seed=seed)
        
        original_print(f"Initial state: {len([p for p in env.unwrapped.all_players if p.stack > 0])} players")
        
        done = False
        steps = 0
        max_steps = 2000  # Much shorter limit for debug
        
        last_state_check = 0
        last_remaining = 0
        stall_counter = 0
        
        while not done and steps < max_steps:
            action_mask = info.get('action_mask', None)
            action = test_agent.act(obs, action_mask=action_mask)
            
            prev_remaining = len([p for p in env.unwrapped.all_players if p.stack > 0])
            
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            remaining = len([p for p in env.unwrapped.all_players if p.stack > 0])
            eliminated = len(env.unwrapped.elimination_order)
            
            # Check for progress every 100 steps
            if steps % 100 == 0:
                original_print(f"Step {steps}: {remaining} remaining, {eliminated} eliminated, done={done}")
                
                # Detect stalling
                if remaining == last_remaining:
                    stall_counter += 1
                    if stall_counter >= 5:  # 500 steps with no progress
                        original_print(f"🚨 STALL DETECTED at step {steps}!")
                        original_print(f"Tournament state: {remaining} remaining, {eliminated} eliminated")
                        original_print(f"Active tables: {len(env.unwrapped._get_active_tables())}")
                        original_print(f"Current table: {env.unwrapped.active_table_id}")
                        
                        # Show blind level information
                        original_print(f"Current blind level: {env.unwrapped.current_blind_level + 1}")
                        original_print(f"Hands this level: {env.unwrapped.hands_played_this_level}/{env.unwrapped.hands_per_blind_level}")
                        blind_level = env.unwrapped.blinds_schedule[env.unwrapped.current_blind_level]
                        if len(blind_level) == 2:
                            sb, bb = blind_level
                            original_print(f"Current blinds: {sb}/{bb}")
                        else:
                            sb, bb, ante = blind_level
                            original_print(f"Current blinds: {sb}/{bb}, ante: {ante}")
                        
                        # Get table details
                        if env.unwrapped.active_table_id in env.unwrapped.tables:
                            table = env.unwrapped.tables[env.unwrapped.active_table_id]
                            original_print(f"Table players: {len(table.players)}")
                            original_print(f"Table active count: {table.get_active_player_count()}")
                            original_print(f"Current player idx: {table.game.current_player_idx}")
                            if table.players:
                                current_player = table.players[table.game.current_player_idx] if table.game.current_player_idx < len(table.players) else None
                                if current_player:
                                    original_print(f"Current player: {current_player.name}, stack: {current_player.stack}")
                                
                                # Show all player states
                                original_print("All player states:")
                                for i, player in enumerate(table.players):
                                    original_print(f"  {i}: {player.name}, stack: {player.stack}, in_hand: {player.in_hand}")
                                
                                # Check for inconsistencies
                                zero_stack_players = [p for p in table.players if p.stack == 0]
                                if zero_stack_players:
                                    original_print(f"Players with 0 stack: {[p.name for p in zero_stack_players]}")
                                    original_print(f"In elimination_order: {[p.name for p in env.unwrapped.elimination_order]}")
                        break
                else:
                    stall_counter = 0
                    
                last_remaining = remaining
            
            if truncated:
                original_print(f"Tournament truncated at step {steps}")
                break
        
        original_print(f"Debug finished: steps={steps}, done={done}")
        original_print(f"Final state: {remaining} remaining, {eliminated} eliminated")
        
    except Exception as e:
        builtins.print = original_print  # Restore for error messages
        original_print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        builtins.print = original_print

if __name__ == "__main__":
    main()
