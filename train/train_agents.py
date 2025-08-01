import argparse
import logging
from env.poker_tournament_env import PokerTournamentEnv
from agents.sharky_agent import SharkyAgent
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import os
# Add more agent imports as you implement them
# run with: python -m train.train_agents --agent sharky --timesteps 20000 --eval-episodes 20 --log INFO

def action_mask_fn(env):
    return env.legal_action_mask()

def parse_args():
    parser = argparse.ArgumentParser(description="Poker RL Agent Training")
    parser.add_argument("--agent", type=str, default="sharky", choices=["sharky"], help="Agent to train")
    parser.add_argument("--timesteps", type=int, default=10000, help="Number of training timesteps")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes for evaluation after training")
    parser.add_argument("--log", type=str, default="INFO", help="Logging level")
    parser.add_argument("--load-model", type=str, default=None, help="Path to a saved model to load")
    parser.add_argument("--save-model", type=str, default="sharky_model.zip", help="Filename to save trained model")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    logger = logging.getLogger("train_agents")

    logger.info(f"Training agent: {args.agent}")
    env = PokerTournamentEnv(num_players=9, starting_stack=1000)
    env = ActionMasker(env, action_mask_fn)  # Wrap with action mask support

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
    agent.learn(total_timesteps=args.timesteps)
    logger.info("Training complete.")

    # Save the trained model
    agent.model.save("sharky_model.zip")
    if os.path.exists(args.save_model):
        raise FileExistsError(f"Model file {args.save_model} already exists. Please choose a different version or delete the file.")
    agent.model.save(args.save_model)
    logger.info(f"Model saved to {args.save_model}")

    # Evaluation
    logger.info(f"Evaluating agent for {args.eval_episodes} episodes...")
    total_reward = 0
    for ep in range(args.eval_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            current_player_idx = getattr(env.unwrapped, "current_player_idx", None)
            players = getattr(env.unwrapped, "players", None)
            get_obs_for_player = getattr(env.unwrapped, "get_obs_for_player", None)
            if current_player_idx is None or players is None or get_obs_for_player is None:
                raise AttributeError("Environment missing required attributes for evaluation.")
            player = players[current_player_idx]
            obs_for_player = get_obs_for_player(player)
            action = agent.act(obs_for_player)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            if done:
                break
        logger.info(f"Episode {ep+1}: Reward = {ep_reward}")
        total_reward += ep_reward

if __name__ == "__main__":
    main()