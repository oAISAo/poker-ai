#!/usr/bin/env python3
"""
Sharky Evolution Runner - Convenient script for training and evaluating Sharky versions

Usage:
    # Train a new version from scratch
    python sharky_evolution_runner.py train 1.0.0

    # Train next version using previous as starting point
    python sharky_evolution_runner.py train 1.0.1 --from 1.0.0

    # Evaluate a trained version
    python sharky_evolution_runner.py evaluate 1.0.0

    # Check training stats
    python sharky_evolution_runner.py stats 1.0.0

    # Train all versions 1.0.0 through 1.0.9
    python sharky_evolution_runner.py train-all 1.0

    # Run tournament between multiple versions
    python sharky_evolution_runner.py tournament 1.0.0 1.0.1 1.0.2
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_sharky_evolution import train_sharky_version, evaluate_agent_tournament, run_multi_agent_tournament
from agents.sharky_agent import SharkyAgent
from env.multi_table_tournament_env import MultiTableTournamentEnv


def get_model_path(version: str) -> str:
    """Get the file path for a Sharky model version"""
    return f"models/sharky_evolution/sharky_{version}.zip"


def get_stats_path(version: str) -> str:
    """Get the file path for Sharky training stats"""
    return f"models/sharky_evolution/sharky_{version}_stats.npy"


def model_exists(version: str) -> bool:
    """Check if a model version exists"""
    return os.path.exists(get_model_path(version))


def train_version(version: str, from_version: Optional[str] = None, timesteps: int = 50000):
    """Train a specific Sharky version"""
    print(f"🦈 === Training Sharky {version} ===")
    
    # Determine starting point
    load_from = None
    if from_version:
        load_from = get_model_path(from_version)
        if not os.path.exists(load_from):
            print(f"❌ Starting model not found: {load_from}")
            return False
        print(f"📂 Starting from Sharky {from_version}")
    elif version != "1.0.0":
        # Auto-detect previous version
        version_parts = version.split('.')
        if len(version_parts) == 2:
            # Handle format like "1.0" 
            major, minor = version_parts
            prev_minor = str(int(minor) - 1)
            prev_version = f"{major}.{prev_minor}"
        elif len(version_parts) == 3:
            # Handle format like "1.0.1"
            major, minor, patch = version_parts
            prev_patch = str(int(patch) - 1)
            if int(patch) > 0:
                prev_version = f"{major}.{minor}.{prev_patch}"
            else:
                # If patch is 0, look for previous minor version
                if int(minor) > 0:
                    prev_minor = str(int(minor) - 1)
                    prev_version = f"{major}.{prev_minor}.9"  # Assume .9 is highest
                else:
                    prev_version = "1.0.0"  # First version
        else:
            print(f"❌ Invalid version format: {version}")
            return False
            
        load_from = get_model_path(prev_version)
        if os.path.exists(load_from):
            print(f"📂 Auto-detected starting point: Sharky {prev_version}")
        else:
            print(f"⚠️  No previous version found, training from scratch")
            load_from = None
    else:
        print(f"🆕 Training from scratch")
    
    # Train the version
    try:
        agent = train_sharky_version(
            version=version,
            timesteps=timesteps,
            load_from=load_from
        )
        print(f"✅ Sharky {version} training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def evaluate_version(version: str, num_tournaments: int = 5):
    """Evaluate a specific Sharky version"""
    print(f"🏆 === Evaluating Sharky {version} ===")
    
    model_path = get_model_path(version)
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Create environment
        env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9, min_players_per_table=2)
        
        # Create and load agent
        agent = SharkyAgent(env, version=version)
        if not agent.load(model_path):
            print(f"❌ Failed to load model: {model_path}")
            return False
        
        print(f"✅ Sharky {version} loaded successfully")
        
        # Run evaluation
        results = evaluate_agent_tournament(agent, num_tournaments=num_tournaments)
        
        # Display results
        print(f"\n📈 Evaluation Results ({num_tournaments} tournaments):")
        print(f"  Average Placement: {results['average_placement']:.2f}/18")
        print(f"  Win Rate: {results['win_rate']:.1%}")
        print(f"  Average Reward: {results['average_reward']:.1f}")
        print(f"  Tournament Placements: {results['placements']}")
        
        # Update and save training stats
        agent.training_stats['tournaments_played'] = num_tournaments
        agent.training_stats['average_placement'] = results['average_placement']
        agent.training_stats['win_rate'] = results['win_rate']
        
        np.save(get_stats_path(version), agent.training_stats)
        print(f"💾 Updated training stats saved")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False


def show_stats(version: str):
    """Show training statistics for a version"""
    print(f"📊 === Sharky {version} Training Stats ===")
    
    stats_path = get_stats_path(version)
    if not os.path.exists(stats_path):
        print(f"❌ Stats not found: {stats_path}")
        return False
    
    try:
        stats = np.load(stats_path, allow_pickle=True).item()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return True
    except Exception as e:
        print(f"❌ Failed to load stats: {e}")
        return False


def train_all_versions(base_version: str, start_from: int = 0, end_at: int = 9, timesteps: int = 25000):
    """Train all versions in a series (e.g., 1.0.0 through 1.0.9)"""
    print(f"🦈 === Training All Sharky {base_version}.x Versions ===")
    
    success_count = 0
    failed_versions = []
    
    for i in range(start_from, end_at + 1):
        version = f"{base_version}.{i}"
        print(f"\n--- Training Version {version} ---")
        
        if train_version(version, timesteps=timesteps):
            success_count += 1
            print(f"✅ {version} completed")
        else:
            failed_versions.append(version)
            print(f"❌ {version} failed")
    
    print(f"\n🎯 Training Summary:")
    print(f"  Successful: {success_count}/{end_at - start_from + 1}")
    if failed_versions:
        print(f"  Failed: {', '.join(failed_versions)}")
    
    return len(failed_versions) == 0


def run_tournament_between_versions(versions: List[str], num_tournaments: int = 3):
    """Run tournaments between multiple Sharky versions"""
    print(f"🏟️ === Multi-Version Tournament ===")
    print(f"Versions: {', '.join(versions)}")
    
    # Load all agents
    agents = []
    env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9)
    
    for version in versions:
        model_path = get_model_path(version)
        if not os.path.exists(model_path):
            print(f"❌ Model not found for version {version}: {model_path}")
            return False
        
        agent = SharkyAgent(env, version=version)
        if agent.load(model_path):
            agents.append(agent)
            print(f"✅ Loaded Sharky {version}")
        else:
            print(f"❌ Failed to load Sharky {version}")
            return False
    
    if len(agents) < 2:
        print(f"❌ Need at least 2 agents for tournament")
        return False
    
    try:
        # Run tournament
        results = run_multi_agent_tournament(agents, num_tournaments=num_tournaments)
        return True
    except Exception as e:
        print(f"❌ Tournament failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Sharky Evolution Runner')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a specific version')
    train_parser.add_argument('version', help='Version to train (e.g., 1.0.0)')
    train_parser.add_argument('--from', dest='from_version', help='Starting version to load from')
    train_parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a specific version')
    eval_parser.add_argument('version', help='Version to evaluate')
    eval_parser.add_argument('--tournaments', type=int, default=5, help='Number of evaluation tournaments')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show training statistics')
    stats_parser.add_argument('version', help='Version to show stats for')
    
    # Train-all command
    train_all_parser = subparsers.add_parser('train-all', help='Train all versions in a series')
    train_all_parser.add_argument('base_version', help='Base version (e.g., 1.0 for 1.0.0-1.0.9)')
    train_all_parser.add_argument('--start', type=int, default=0, help='Starting number')
    train_all_parser.add_argument('--end', type=int, default=9, help='Ending number')
    train_all_parser.add_argument('--timesteps', type=int, default=25000, help='Training timesteps per version')
    
    # Tournament command
    tournament_parser = subparsers.add_parser('tournament', help='Run tournament between versions')
    tournament_parser.add_argument('versions', nargs='+', help='Versions to compete')
    tournament_parser.add_argument('--tournaments', type=int, default=3, help='Number of tournaments')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'train':
        train_version(args.version, args.from_version, args.timesteps)
    
    elif args.command == 'evaluate':
        evaluate_version(args.version, args.tournaments)
    
    elif args.command == 'stats':
        show_stats(args.version)
    
    elif args.command == 'train-all':
        train_all_versions(args.base_version, args.start, args.end, args.timesteps)
    
    elif args.command == 'tournament':
        run_tournament_between_versions(args.versions, args.tournaments)


if __name__ == "__main__":
    main()
