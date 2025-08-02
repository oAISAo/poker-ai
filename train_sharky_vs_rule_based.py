#!/usr/bin/env python3
"""
Train Sharky against rule-based opponents instead of random players
This should provide much better training quality and more meaningful evolution
"""

import os
import sys
import builtins
import numpy as np
from typing import Dict, List, Optional
# Create vectorized environment (set very high episode limit for tournaments)
from gymnasium.wrappers import TimeLimit
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.rule_based_tournament_env import create_rule_based_training_env
from agents.sharky_agent import SharkyAgent
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor


def action_mask_fn(env):
    """Action mask function for the environment"""
    return env.legal_action_mask()


def train_sharky_vs_rule_based(
    version: str,
    timesteps: int = 50000,
    load_from: Optional[str] = None,
    save_path: Optional[str] = None
) -> SharkyAgent:
    """
    Train Sharky agent against rule-based opponents
    
    Args:
        version: Version string (e.g., "1.0.1")
        timesteps: Number of training timesteps
        load_from: Path to previous model to load from
        save_path: Path to save trained model (auto-generated if None)
    
    Returns:
        Trained SharkyAgent
    """
    
    print(f"🦈 Training Sharky {version} vs Rule-Based Opponents")
    print(f"📊 Training setup:")
    print(f"  • Timesteps: {timesteps:,}")
    print(f"  • Opponents: 4 TAG, 4 LAG, 4 Rock, 5 Fish bots")
    print(f"  • Tournament: Turbo format (9-hand blind levels)")
    print(f"  • Blind structure: 10/20 → 20/40 → 30/60 → 40/80 → 50/100...")
    print(f"  • Starting from: {load_from if load_from else 'scratch'}")
    
    # Suppress debug spam during training
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
            'wins', 'chips from pot', 'SB stack:', 'BB stack:', 'Removed', 'eliminated players',
            '[PLAYER bet_chips]','[DEBUG', '[BALANCE_TABLE]', '[INCONSISTENCY-CHECK]', '[SHOWDOWN]', '[PLAYER', # aisa comment out when debugging
            'Error in game step',
        ]):
            pass  # Skip these debug messages
        else:
            original_print(*args, **kwargs)
    
    # Replace print function to suppress debug spam
    builtins.print = quiet_print
    
    # Create training environment with rule-based opponents
    def make_env():
        env = create_rule_based_training_env(
            total_players=18
        )
        # Wrap with action masker for MaskablePPO
        env = ActionMasker(env, action_mask_fn)
        return env
    
    def make_env_with_limit():
        env = create_rule_based_training_env(total_players=18)
        env = ActionMasker(env, action_mask_fn)
        # Set much higher time limit for tournaments (20000 steps should allow completion)
        env = TimeLimit(env, max_episode_steps=20000)
        return env
    
    env = make_vec_env(make_env_with_limit, n_envs=1)
    env = VecMonitor(env)
    
    # Create or load model
    if load_from and os.path.exists(load_from):
        print(f"📂 Loading from {load_from}")
        model = MaskablePPO.load(load_from, env=env, device='cpu')
        print(f"✅ Model loaded successfully")
    else:
        print("🆕 Creating new model from scratch")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
            ),
            device='cpu',
            verbose=1
        )
    
    # Train the model
    print(f"🏋️ Starting training for {timesteps:,} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=False if load_from else True,
        progress_bar=True
    )
    
    # Save the trained model
    if save_path is None:
        os.makedirs("models/sharky_evolution", exist_ok=True)
        save_path = f"models/sharky_evolution/sharky_{version}.zip"
    
    model.save(save_path)
    print(f"💾 Model saved to {save_path}")
    
    # Create SharkyAgent wrapper
    test_env = create_rule_based_training_env(total_players=18)
    agent = SharkyAgent(test_env, version=version)
    agent.model = model
    
    # Initialize training stats
    agent.training_stats = {
        'version': version,
        'timesteps_trained': timesteps,
        'opponent_types': 'Rule-based (TAG/LAG/Rock/Fish)',
        'training_complete': True,
        'model_path': save_path
    }
    
    # Save training stats
    stats_path = f"models/sharky_evolution/sharky_{version}_stats.npz"
    np.savez(stats_path, **agent.training_stats)
    
    # Restore original print function
    builtins.print = original_print
    
    print(f"✅ Training completed! Sharky {version} ready for evaluation.")
    
    return agent


def evaluate_vs_rule_based(agent: SharkyAgent, num_tournaments: int = 5) -> Dict[str, object]:
    """
    Evaluate agent against rule-based opponents
    """
    print(f"🏆 Evaluating {agent.version} vs rule-based opponents...")
    
    # Create evaluation environment
    env = create_rule_based_training_env(total_players=18)
    
    placements = []
    rewards = []
    
    for tournament_num in range(num_tournaments):
        obs, info = env.reset()
        done = False
        tournament_reward = 0
        steps = 0
        
        while not done and steps < 3000:
            action_mask = info.get('action_mask', None)
            action = agent.act(obs, action_mask=action_mask, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            tournament_reward += reward
            steps += 1
            
            if truncated:
                break
        
        # Calculate placement
        remaining_players = len([p for p in env.all_players if p.stack > 0])
        eliminated_players = len(env.elimination_order)
        
        if remaining_players == 1:
            placement = 1 if tournament_reward > 1000 else eliminated_players + 1
        else:
            placement = eliminated_players + 1
        
        placements.append(placement)
        rewards.append(tournament_reward)
        
        print(f"  Tournament {tournament_num + 1}: Placement {placement}, Reward {tournament_reward:.1f}")
    
    # Calculate statistics
    avg_placement = np.mean(placements)
    win_rate = len([p for p in placements if p == 1]) / num_tournaments
    avg_reward = np.mean(rewards)
    
    results = {
        'average_placement': avg_placement,
        'win_rate': win_rate,
        'average_reward': avg_reward,
        'placements': placements,
        'rewards': rewards
    }
    
    print(f"📊 Results vs Rule-Based: Avg Placement: {avg_placement:.2f}, Win Rate: {win_rate:.1%}, Avg Reward: {avg_reward:.1f}")
    
    return results


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train Sharky vs Rule-Based Opponents')
    parser.add_argument('version', help='Version to train (e.g., 1.0.1)')
    parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps')
    parser.add_argument('--from', dest='load_from', help='Load from previous model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    
    args = parser.parse_args()
    
    # Train the agent
    agent = train_sharky_vs_rule_based(
        version=args.version,
        timesteps=args.timesteps,
        load_from=args.load_from
    )
    
    # Evaluate if requested
    if args.evaluate:
        evaluate_vs_rule_based(agent, num_tournaments=5)
