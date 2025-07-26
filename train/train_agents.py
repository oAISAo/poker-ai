import argparse
import logging
from env.poker_tournament_env import PokerTournamentEnv
from agents.sharky_agent import SharkyAgent
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
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
        agent = SharkyAgent(env, use_maskable_ppo=True)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # Training
    logger.info(f"Starting training for {args.timesteps} timesteps...")
    agent.learn(total_timesteps=args.timesteps)
    logger.info("Training complete.")

    # Evaluation
    logger.info(f"Evaluating agent for {args.eval_episodes} episodes...")
    total_reward = 0
    for ep in range(args.eval_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        while not done:
            current_player_idx = env.unwrapped.current_player_idx
            player = env.unwrapped.players[current_player_idx]
            obs_for_player = env.unwrapped.get_obs_for_player(player)
            action = agent.act(obs_for_player)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            if done:
                break
        logger.info(f"Episode {ep+1}: Reward = {ep_reward}")
        total_reward += ep_reward

if __name__ == "__main__":
    main()