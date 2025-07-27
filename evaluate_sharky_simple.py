#!/usr/bin/env python3
"""
Simple Sharky Evaluation Script
Evaluates a Sharky version with minimal debug output
"""

import os
import sys
import numpy as np
import builtins

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_sharky_simple.py <version>")
        sys.exit(1)
    
    version = sys.argv[1]
    
    print(f'🏆 Evaluating Sharky {version} Performance...')
    
    # Save original print and create a filter for debug spam
    original_print = builtins.print
    
    def quiet_print(*args, **kwargs):
        text = ' '.join(str(arg) for arg in args)
        # Only block the most spammy debug messages
        if any(spam in text for spam in [
            '[DEBUG]', 'Player_', 'posts ante', 'posts small blind', 'posts big blind',
            'was dealt:', 'Community cards dealt:', 'checks.', 'calls.', 'raises.', 'folds.',
            'Stack:', 'CurrentBet:', 'Pot:', 'ToCall:', 'RaiseTo:', 'Fixed game state',
            'Advancing to phase:', 'Removing', 'from players_to_act', 'handle_',
            '[WARNING] Pot mismatch:', '--- Showdown ---', 'Blinds increased to',
            'wins', 'chips from pot', 'SB stack:', 'BB stack:',
            'Tournament initialized:', 'Using cpu device', 'Wrapping the env',
            '📂 Sharky', 'loaded from models'
        ]):
            return  # Block these
        original_print(*args, **kwargs)
    
    try:
        from agents.sharky_agent import SharkyAgent
        from env.multi_table_tournament_env import MultiTableTournamentEnv
        
        # Apply quiet mode after imports
        builtins.print = quiet_print
        
        model_path = f'models/sharky_evolution/sharky_{version}.zip'
        
        if not os.path.exists(model_path):
            original_print(f'❌ Model file not found: {model_path}')
            return
        
        # Create environment
        env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9, min_players_per_table=6)
        
        # Create agent
        sharky = SharkyAgent(env, version=version)
        
        if not sharky.load(model_path):
            original_print('❌ Model loading failed')
            return
        
        original_print(f'✅ Sharky {version} loaded successfully')
        original_print('\n🏆 Running 5 evaluation tournaments...')
        
        placements = []
        rewards = []
        
        for i in range(5):
            obs, info = env.reset()
            done = False
            tournament_reward = 0
            steps = 0
            
            while not done and steps < 3000:
                action_mask = info.get('action_mask', None)
                action = sharky.act(obs, action_mask=action_mask, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                tournament_reward += reward
                steps += 1
                
                if truncated:
                    break
            
            placement = 1 if tournament_reward > 1000 else 18
            placements.append(placement)
            rewards.append(tournament_reward)
            
            original_print(f'Tournament {i+1}: Placement {placement}, Reward {tournament_reward:.1f}')
        
        # Results
        avg_placement = np.mean(placements)
        win_rate = len([p for p in placements if p == 1]) / 5
        avg_reward = np.mean(rewards)
        
        original_print(f'\n📊 Results: Avg Placement: {avg_placement:.2f}, Win Rate: {win_rate:.1%}, Avg Reward: {avg_reward:.1f}')
        
        original_print(f'\n📈 Evaluation Results (5 tournaments):')
        original_print(f'  Average Placement: {avg_placement:.2f}/18')
        original_print(f'  Win Rate: {win_rate:.1%}')
        original_print(f'  Average Reward: {avg_reward:.1f}')
        original_print(f'  Tournament Placements: {placements}')
        original_print(f'  Individual Rewards: {[round(r, 1) for r in rewards]}')
        
        original_print(f'\n🔍 Analysis:')
        wins = len([p for p in placements if p == 1])
        original_print(f'  Tournaments Won: {wins}/5')
        original_print(f'  Best Placement: {min(placements)}')
        original_print(f'  Worst Placement: {max(placements)}')
        
        # Save stats
        sharky.training_stats['tournaments_played'] = 5
        sharky.training_stats['average_placement'] = avg_placement  
        sharky.training_stats['win_rate'] = win_rate
        np.save(f'models/sharky_evolution/sharky_{version}_stats.npy', sharky.training_stats)
        original_print('\n💾 Updated training stats saved')
        
    except Exception as e:
        builtins.print = original_print  # Restore for error messages
        original_print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        builtins.print = original_print

if __name__ == "__main__":
    main()
