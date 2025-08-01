import argparse
import logging
from env.multi_table_tournament_env import MultiTableTournamentEnv
from agents.sharky_agent import SharkyAgent
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import os
import numpy as np

# Multi-table tournament training script
# Run with: python -m train.train_multi_table_agents --agent sharky --timesteps 50000 --tournaments 5 --log INFO

def action_mask_fn(env):
    return env.legal_action_mask()

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Table Poker Tournament RL Agent Training")
    parser.add_argument("--agent", type=str, default="sharky", choices=["sharky"], help="Agent to train")
    parser.add_argument("--timesteps", type=int, default=50000, help="Number of training timesteps")
    parser.add_argument("--tournaments", type=int, default=3, help="Number of tournaments for evaluation")
    parser.add_argument("--total-players", type=int, default=27, help="Total players in tournament")
    parser.add_argument("--max-per-table", type=int, default=9, help="Maximum players per table")
    parser.add_argument("--hands-per-level", type=int, default=9, help="Hands per blind level")
    parser.add_argument("--log", type=str, default="INFO", help="Logging level")
    parser.add_argument("--load-model", type=str, default=None, help="Path to a saved model to load")
    parser.add_argument("--save-model", type=str, default="multi_table_sharky_model.zip", help="Filename to save trained model")
    return parser.parse_args()

def evaluate_tournament_performance(env, agent, num_tournaments=3):
    """Evaluate agent performance across multiple tournaments"""
    results = []
    
    for tournament_num in range(num_tournaments):
        logging.info(f"Running evaluation tournament {tournament_num + 1}/{num_tournaments}")
        
        obs, info = env.reset()
        done = False
        steps = 0
        total_reward = 0
        initial_players = env.total_players
        
        while not done and steps < 10000:  # Limit steps to prevent infinite loops
            # Get current player and their observation
            if env.active_table_id in env.tables:
                table = env.tables[env.active_table_id]
                if table.players and table.game.current_player_idx < len(table.players):
                    current_player = table.players[table.game.current_player_idx]
                    
                    # For evaluation, we'll use the agent for all players
                    action = agent.act(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    steps += 1
                else:
                    # No valid player, skip
                    break
            else:
                # No active table
                break
        
        # Tournament results
        remaining_players = len([p for p in env.all_players if p.stack > 0])
        eliminated_players = len(env.elimination_order)
        
        tournament_result = {
            "tournament": tournament_num + 1,
            "initial_players": initial_players,
            "remaining_players": remaining_players,
            "eliminated_players": eliminated_players,
            "total_reward": total_reward,
            "steps": steps,
            "final_stats": env.get_tournament_stats()
        }
        
        results.append(tournament_result)
        
        # Log tournament summary
        stats = env.get_tournament_stats()
        logging.info(f"Tournament {tournament_num + 1} completed:")
        logging.info(f"  Players remaining: {stats['remaining_players']}")
        logging.info(f"  Active tables: {stats['active_tables']}")
        logging.info(f"  Blind level reached: {stats['current_blind_level']}")
        logging.info(f"  Total hands: {stats['hands_played']}")
        logging.info(f"  Total reward: {total_reward}")
    
    return results

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    logger = logging.getLogger("train_multi_table_agents")

    logger.info(f"Training multi-table tournament agent: {args.agent}")
    logger.info(f"Tournament config: {args.total_players} players, {args.max_per_table} max per table")
    
    # Create multi-table tournament environment
    env = MultiTableTournamentEnv(
        total_players=args.total_players,
        max_players_per_table=args.max_per_table,
        hands_per_blind_level=args.hands_per_level,
        table_balancing_threshold=6
    )
    
    # Wrap with action mask support
    env = ActionMasker(env, action_mask_fn)

    # Agent selection
    if args.agent == "sharky":
        agent = SharkyAgent(env)
        if args.load_model:
            agent.model = MaskablePPO.load(args.load_model, env=env)
            logger.info(f"Loaded model from {args.load_model}")
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # Training
    logger.info(f"Starting training for {args.timesteps} timesteps...")
    logger.info("Note: Multi-table training may take longer due to complex environment")
    
    try:
        agent.learn(total_timesteps=args.timesteps)
        logger.info("Training complete.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    # Save the trained model
    if os.path.exists(args.save_model):
        logger.warning(f"Model file {args.save_model} already exists. Overwriting...")
    
    agent.model.save(args.save_model)
    logger.info(f"Model saved to {args.save_model}")

    # Evaluation across multiple tournaments
    logger.info(f"Evaluating agent across {args.tournaments} tournaments...")
    
    try:
        results = evaluate_tournament_performance(env.unwrapped, agent, args.tournaments)
        
        # Calculate aggregate statistics
        total_tournaments = len(results)
        avg_remaining = np.mean([r["remaining_players"] for r in results])
        avg_eliminated = np.mean([r["eliminated_players"] for r in results])
        avg_reward = np.mean([r["total_reward"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])
        
        logger.info("=== Evaluation Summary ===")
        logger.info(f"Tournaments completed: {total_tournaments}")
        logger.info(f"Average players remaining: {avg_remaining:.1f}")
        logger.info(f"Average players eliminated: {avg_eliminated:.1f}")
        logger.info(f"Average total reward: {avg_reward:.2f}")
        logger.info(f"Average steps per tournament: {avg_steps:.0f}")
        
        # Show individual tournament results
        for result in results:
            stats = result["final_stats"]
            logger.info(f"Tournament {result['tournament']}: "
                       f"{stats['remaining_players']} players left, "
                       f"Level {stats['current_blind_level']}, "
                       f"{stats['hands_played']} hands, "
                       f"Reward: {result['total_reward']:.2f}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
