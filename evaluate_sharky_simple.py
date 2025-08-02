#!/usr/bin/env python3
"""
Simple Sharky Evaluation Script
Evaluates a Sharky version with minimal debug output
"""

import os
import sys
import numpy as np
import builtins
import random
import time
import traceback
from agents.sharky_agent import SharkyAgent
from env.rule_based_tournament_env import create_rule_based_training_env

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_sharky_simple.py <version>")
        #sys.exit(1)
    
    version = sys.argv[1]
    
    print(f'🏆 Evaluating Sharky {version} Performance...')
    
    # Save original print and create a filter for debug spam
    original_print = builtins.print
    
    def quiet_print(*args, **kwargs):
        text = ' '.join(str(arg) for arg in args)
        # Only block the most spammy debug messages
        if any(spam in text for spam in [
            'posts ante', 'posts small blind', 'posts big blind',
            'was dealt:', 'Community cards dealt:', 'checks.', 'calls.', 'raises.', 'folds.',
            'Stack:', 'CurrentBet:', 'Pot:', 'ToCall:', 'RaiseTo:', 'Fixed game state',
            'Advancing to phase:', 'Removing', 'from players_to_act', 'handle_',
            '--- Showdown ---', 'Blinds increased to',
            'wins', 'chips from pot', 'SB stack:', 'BB stack:', 'Exiting step:',
            'Tournament initialized:', 'Using cpu device', 'Wrapping the env', 'Entering step:',
            '📂 Sharky', 'loaded from models', "==> ", "'s turn:", "Action=", 'bets ', 'raises to',
            '[PLAYER bet_chips]','[DEBUG', '[BALANCE_TABLE]', '[INCONSISTENCY-CHECK]', '[SHOWDOWN]', '[PLAYER', # aisa comment out when debugging
            'bet_chips', 'suppress_log'
        ]):
            return  # Block these
        original_print(*args, **kwargs)
    
    try:
        # Apply quiet mode after imports
        builtins.print = quiet_print
        
        # Handle both version number and full path inputs
        if version.endswith('.zip') and os.path.exists(version):
            # Full path provided
            model_path = version
            # Extract version from path for display
            version_display = os.path.basename(version).replace('sharky_', '').replace('.zip', '')
        else:
            # Version number provided
            model_path = f'models/sharky_evolution/sharky_{version}.zip'
            version_display = version
        
        if not os.path.exists(model_path):
            original_print(f'❌ Model file not found: {model_path}')
            return
        
        # Create environment
        env = create_rule_based_training_env(total_players=18)
        
        # Create agent
        sharky = SharkyAgent(env, version=version_display)
        
        if not sharky.load(model_path):
            original_print('❌ Model loading failed')
            return
        
        original_print(f'✅ Sharky {version_display} loaded successfully')
        original_print('\n🏆 Running 1 debug tournament...')
        
        placements = []
        rewards = []
        

        for i in range(1):  # Just 1 tournament for debugging
            original_print(f'\n=== TOURNAMENT {i+1} START ===')
            # Use different random seed for each tournament (valid 32-bit range)
            seed = (int(time.time()) + i * 1000 + random.randint(0, 999)) % (2**32 - 1)
            original_print(f'Tournament {i+1} seed: {seed}')
            obs, info = env.reset(seed=seed)

            # Robustly access all custom attributes
            custom_env = env.unwrapped
            all_players = getattr(custom_env, "all_players", None)
            elimination_order = getattr(custom_env, "elimination_order", None)
            total_players = getattr(custom_env, "total_players", None)
            get_placement_reward = getattr(custom_env, "_get_placement_reward", None)
            if all_players is None:
                raise AttributeError("env.unwrapped does not have 'all_players' attribute")
            if elimination_order is None:
                raise AttributeError("env.unwrapped does not have 'elimination_order' attribute")
            if total_players is None:
                raise AttributeError("env.unwrapped does not have 'total_players' attribute")
            if get_placement_reward is None:
                raise AttributeError("env.unwrapped does not have '_get_placement_reward' method")

            # Log initial player names
            player_names = [p.name for p in all_players]
            original_print(f'Tournament {i+1} players: {player_names[:5]}...') # Show first 5
            original_print(f'Player_0 position: {[p.name for p in all_players].index("Player_0") if "Player_0" in [p.name for p in all_players] else "NOT FOUND"}')
            original_print(f'all_players[0].name: {all_players[0].name}')

            done = False
            tournament_reward = 0
            steps = 0

            while not done and steps < 15000:
                action_mask = info.get('action_mask', None)
                action = sharky.act(obs, action_mask=action_mask, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                tournament_reward += reward
                steps += 1

                # Log every 500 steps to see tournament progress
                if steps % 500 == 0:
                    remaining = len([p for p in all_players if p.stack > 0])
                    eliminated = len(elimination_order)
                    original_print(f'Step {steps}: {remaining} remaining, {eliminated} eliminated, done={done}, truncated={truncated}')

                if truncated:
                    original_print(f'Tournament truncated at step {steps}')
                    break
            
            # Check why we exited the loop
            if steps >= 15000:
                original_print(f'Tournament hit step limit (15000)')
                truncated = True  # Manually set truncated since we hit the limit
            
            original_print(f'Tournament {i+1} finished after {steps} steps, done={done}, truncated={truncated}')
            
            # Log final tournament state details
            remaining_stacks = [(p.name, p.stack) for p in all_players if p.stack > 0]
            eliminated_stacks = [(p.name, p.stack) for p in all_players if p.stack == 0]
            original_print(f'Players with stacks: {len(remaining_stacks)}, eliminated: {len(eliminated_stacks)}')
            original_print(f'Elimination order length: {len(elimination_order)}')
            if len(remaining_stacks) <= 5:
                original_print(f'Remaining players: {remaining_stacks}')
            if len(eliminated_stacks) <= 5:
                original_print(f'Eliminated players: {eliminated_stacks}')

            # Calculate placement based on tournament state
            remaining_players = len([p for p in all_players if p.stack > 0])
            eliminated_players = len(elimination_order)

            # Find the actual Player_0 (Sharky agent)
            agent_player = None
            for player in all_players:
                if player.name == "Player_0":
                    agent_player = player
                    break

            if agent_player is None:
                original_print(f'ERROR: Player_0 not found in tournament!')
                placement = 18  # Worst case fallback
            else:
                original_print(f'Tournament {i+1} final state: {remaining_players} remaining, {eliminated_players} eliminated')
                original_print(f'Agent {agent_player.name} stack: {agent_player.stack}, in elimination order: {agent_player in elimination_order}')

                if agent_player in elimination_order:
                    elimination_pos = elimination_order.index(agent_player) + 1
                    original_print(f'Agent elimination position: {elimination_pos}')

                if remaining_players == 1:
                    # Tournament ended, check if our agent won
                    if agent_player.stack > 0:  # Agent survived = winner
                        placement = 1
                        original_print(f'Agent won tournament (still has stack: {agent_player.stack})')
                    else:
                        # Agent was eliminated, calculate placement from elimination order
                        if agent_player in elimination_order:
                            elimination_position = elimination_order.index(agent_player) + 1
                            placement = total_players - elimination_position + 1
                            original_print(f'Agent eliminated at position {elimination_position}, placement = {placement}')
                        else:
                            placement = eliminated_players + 1  # Fallback
                            original_print(f'Agent not in elimination order, using fallback placement = {placement}')
                else:
                    # Tournament still running, agent was eliminated
                    if agent_player in elimination_order:
                        elimination_position = elimination_order.index(agent_player) + 1
                        placement = total_players - elimination_position + 1
                        original_print(f'Tournament truncated, agent eliminated at position {elimination_position}, placement = {placement}')
                    else:
                        # Agent has 0 stack but not in elimination order - this indicates a bug
                        if agent_player.stack == 0:
                            placement = total_players  # Last place
                            original_print(f'Agent has 0 stack but not in elimination order - assigning last place = {placement}')
                        else:
                            placement = eliminated_players + 1  # Fallback
                            original_print(f'Tournament truncated, agent not in elimination order, using fallback placement = {placement}')
            
            placements.append(placement)
            rewards.append(tournament_reward)
            
            # Extract just the placement reward for clearer understanding
            if placement == 1:
                placement_reward_only = get_placement_reward(1)
            else:
                placement_reward_only = get_placement_reward(placement)
            
            original_print(f'Tournament {i+1}: Placement {placement}, Total Reward {tournament_reward:.1f} (Placement reward: {placement_reward_only})')
            original_print(f'=== TOURNAMENT {i+1} END ===\n')
        
        # Results
        avg_placement = np.mean(placements)
        win_rate = len([p for p in placements if p == 1]) / 1  # Changed from 5 to 1
        avg_reward = np.mean(rewards)
        
        original_print(f'\n📊 Results: Avg Placement: {avg_placement:.2f}, Win Rate: {win_rate:.1%}, Avg Reward: {avg_reward:.1f}')
        
        original_print(f'\n📈 Evaluation Results (1 tournament):')
        original_print(f'  Average Placement: {avg_placement:.2f}/18')
        original_print(f'  Win Rate: {win_rate:.1%}')
        original_print(f'  Average Reward: {avg_reward:.1f}')
        original_print(f'  Tournament Placements: {placements}')
        original_print(f'  Individual Rewards: {[round(r, 1) for r in rewards]}')
        
        original_print(f'\n🔍 Analysis:')
        wins = len([p for p in placements if p == 1])
        original_print(f'  Tournaments Won: {wins}/1')
        original_print(f'  Best Placement: {min(placements)}')
        original_print(f'  Worst Placement: {max(placements)}')
        
        # Save stats
        sharky.training_stats['tournaments_played'] = 1  # Changed from 5 to 1
        sharky.training_stats['average_placement'] = float(avg_placement)
        sharky.training_stats['win_rate'] = float(win_rate)
        # Save as .npz for dict compatibility
        np.savez(f'models/sharky_evolution/sharky_{version_display}_stats.npz', **sharky.training_stats)
        original_print('\n💾 Updated training stats saved')
        
    except Exception as e:
        builtins.print = original_print  # Restore for error messages
        original_print(f'❌ Error: {e}')
        traceback.print_exc()
    finally:
        builtins.print = original_print

if __name__ == "__main__":
    main()
