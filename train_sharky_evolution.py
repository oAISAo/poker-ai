#!/usr/bin/env python3
"""
Sharky Evolution Training Script - Phase 1

This script implements the Sharky self-play evolution process:
1. Train Sharky from scratch → save as sharky_1.0.0
2. Continue training → save as sharky_1.0.1, ..., sharky_1.0.9  
3. Run tournament with all versions
4. Select best performer → save as sharky_1.1.0

Usage:
    python train_sharky_evolution.py --phase 1 --version 1.0.0
    python train_sharky_evolution.py --phase tournament --generation 1.0
"""

import os
import argparse
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Any

# Environment and agent imports
from env.multi_table_tournament_env import MultiTableTournamentEnv
from agents.sharky_agent import SharkyAgent, TournamentCallback
from sb3_contrib.common.wrappers import ActionMasker

def action_mask_fn(env):
    """Action mask function for the environment"""
    return env.legal_action_mask()

def create_training_environment(total_players=18, hands_per_level=25):
    """Create the multi-table tournament environment for training"""
    env = MultiTableTournamentEnv(
        total_players=total_players,
        max_players_per_table=9,
        starting_stack=1000,
        hands_per_blind_level=hands_per_level,
        table_balancing_threshold=6
    )
    
    # Wrap with action masker for MaskablePPO
    env = ActionMasker(env, action_mask_fn)
    return env

def train_sharky_version(version: str, timesteps: int = 50000, 
                        load_from: str = None) -> SharkyAgent:
    """Train a single Sharky version"""
    print(f"\n🦈 === Training Sharky {version} ===")
    print(f"Timesteps: {timesteps:,}")
    print(f"Load from: {load_from if load_from else 'Training from scratch'}")
    
    # Create environment
    env = create_training_environment()
    
    # Create agent
    agent = SharkyAgent(env, name="Sharky", version=version, verbose=1)
    
    # Load previous model if specified
    if load_from and os.path.exists(load_from):
        print(f"📂 Loading previous model: {load_from}")
        agent.load(load_from)
    
    # Create callback for monitoring
    callback = TournamentCallback(verbose=1)
    
    # Train the agent
    start_time = time.time()
    agent.learn(total_timesteps=timesteps, callback=callback)
    training_time = time.time() - start_time
    
    # Save the trained model
    model_dir = "models/sharky_evolution"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"sharky_{version}.zip")
    agent.save(model_path)
    
    print(f"✅ Training completed in {training_time:.1f} seconds")
    print(f"💾 Model saved: {model_path}")
    
    return agent

def evaluate_agent_tournament(agent: SharkyAgent, num_tournaments: int = 5) -> Dict[str, float]:
    """Evaluate an agent's performance in tournaments"""
    print(f"\n🏆 Evaluating {agent.get_name()} over {num_tournaments} tournaments...")
    
    env = create_training_environment()
    placements = []
    rewards = []
    
    for tournament_num in range(num_tournaments):
        obs, info = env.reset()
        done = False
        tournament_reward = 0
        steps = 0
        max_steps = 5000  # Prevent infinite loops
        
        while not done and steps < max_steps:
            # Get action mask
            action_mask = info.get('action_mask', None)
            
            # Get action from agent
            action = agent.act(obs, action_mask=action_mask, deterministic=True)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            tournament_reward += reward
            steps += 1
            
            if truncated:
                break
        
        # Calculate placement
        remaining_players = len([p for p in env.unwrapped.all_players if p.stack > 0])
        eliminated_players = len(env.unwrapped.elimination_order)
        
        if remaining_players == 1:
            # Find if our agent won (simplified - assumes single agent evaluation)
            placement = 1 if tournament_reward > 100 else eliminated_players + 1
        else:
            placement = eliminated_players + 1
        
        placements.append(placement)
        rewards.append(tournament_reward)
        
        print(f"  Tournament {tournament_num + 1}: Placement {placement}, Reward {tournament_reward:.1f}")
    
    # Calculate statistics
    avg_placement = np.mean(placements)
    win_rate = len([p for p in placements if p == 1]) / num_tournaments
    avg_reward = np.mean(rewards)
    
    stats = {
        'average_placement': avg_placement,
        'win_rate': win_rate,
        'average_reward': avg_reward,
        'placements': placements,
        'rewards': rewards
    }
    
    print(f"📊 Results: Avg Placement: {avg_placement:.2f}, Win Rate: {win_rate:.1%}, Avg Reward: {avg_reward:.1f}")
    
    return stats

def run_multi_agent_tournament(agents: List[SharkyAgent], num_tournaments: int = 3) -> Dict[str, Any]:
    """Run tournaments with multiple Sharky versions competing"""
    print(f"\n🏟️  === Multi-Agent Tournament ===")
    print(f"Agents: {[agent.get_name() for agent in agents]}")
    print(f"Tournaments: {num_tournaments}")
    
    # This is a simplified implementation - in a full version, you'd need to 
    # implement multi-agent support in the environment
    results = {}
    
    for agent in agents:
        print(f"\n--- Evaluating {agent.get_name()} ---")
        stats = evaluate_agent_tournament(agent, num_tournaments)
        results[agent.get_name()] = stats
    
    # Find best performer
    best_agent = min(agents, key=lambda a: results[a.get_name()]['average_placement'])
    
    print(f"\n🏆 Tournament Results Summary:")
    for agent_name, stats in results.items():
        print(f"  {agent_name}: Avg Placement {stats['average_placement']:.2f}, Win Rate {stats['win_rate']:.1%}")
    
    print(f"\n👑 Best Performer: {best_agent.get_name()}")
    
    return {
        'results': results,
        'best_agent': best_agent.get_name(),
        'best_stats': results[best_agent.get_name()]
    }

def main():
    parser = argparse.ArgumentParser(description="Sharky Evolution Training")
    parser.add_argument("--phase", choices=["train", "tournament", "evolve"], default="train",
                       help="Training phase: train single version, run tournament, or full evolution")
    parser.add_argument("--version", type=str, default="1.0.0", 
                       help="Version to train (e.g., '1.0.0')")
    parser.add_argument("--generation", type=str, default="1.0",
                       help="Generation for tournament (e.g., '1.0' for versions 1.0.0-1.0.9)")
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Training timesteps per version")
    parser.add_argument("--load-from", type=str, default=None,
                       help="Path to previous model to continue training")
    parser.add_argument("--tournaments", type=int, default=5,
                       help="Number of evaluation tournaments")
    
    args = parser.parse_args()
    
    if args.phase == "train":
        # Train a single Sharky version
        agent = train_sharky_version(
            version=args.version,
            timesteps=args.timesteps,
            load_from=args.load_from
        )
        
        # Quick evaluation
        stats = evaluate_agent_tournament(agent, num_tournaments=3)
        print(f"\n📈 Quick evaluation results:")
        print(f"Average placement: {stats['average_placement']:.2f}")
        print(f"Win rate: {stats['win_rate']:.1%}")
        
    elif args.phase == "tournament":
        # Load all versions of a generation and run tournament
        model_dir = "models/sharky_evolution"
        agents = []
        
        for i in range(10):  # versions .0 through .9
            version = f"{args.generation}.{i}"
            model_path = os.path.join(model_dir, f"sharky_{version}.zip")
            
            if os.path.exists(model_path):
                env = create_training_environment()
                agent = SharkyAgent(env, version=version, verbose=0)
                if agent.load(model_path):
                    agents.append(agent)
                    print(f"✅ Loaded Sharky {version}")
                else:
                    print(f"❌ Failed to load Sharky {version}")
            else:
                print(f"⚠️  Model not found: {model_path}")
        
        if agents:
            tournament_results = run_multi_agent_tournament(agents, args.tournaments)
            
            # Save results
            results_dir = "results/sharky_evolution"
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(results_dir, f"tournament_{args.generation}_{timestamp}.npy")
            np.save(results_path, tournament_results)
            print(f"💾 Results saved: {results_path}")
        else:
            print("❌ No agents found for tournament")
    
    elif args.phase == "evolve":
        # Full evolution process: train versions 1.0.0 through 1.0.9
        print(f"\n🧬 === Sharky Evolution Process ===")
        print(f"Training generation {args.generation}")
        
        model_dir = "models/sharky_evolution"
        os.makedirs(model_dir, exist_ok=True)
        
        for i in range(10):  # Train versions .0 through .9
            version = f"{args.generation}.{i}"
            
            # For subsequent versions, load from previous
            load_from = None
            if i > 0:
                prev_version = f"{args.generation}.{i-1}"
                load_from = os.path.join(model_dir, f"sharky_{prev_version}.zip")
            
            # Train this version
            agent = train_sharky_version(
                version=version,
                timesteps=args.timesteps,
                load_from=load_from
            )
            
            # Quick evaluation
            stats = evaluate_agent_tournament(agent, num_tournaments=2)
            print(f"Sharky {version} - Avg Placement: {stats['average_placement']:.2f}")
        
        print(f"\n✅ Evolution training complete!")
        print(f"🏟️  Now run: python train_sharky_evolution.py --phase tournament --generation {args.generation}")

if __name__ == "__main__":
    main()
