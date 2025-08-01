import os
import numpy as np
from typing import Optional, Dict, Union
from agents.base_rl_agent import BaseRLAgent
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch

class SharkyAgent(BaseRLAgent):
    """
    Sharky - Deep RL agent for multi-table poker tournaments
    Uses MaskablePPO for action masking and tournament-specific learning
    """
    
    def __init__(self, env, name="Sharky", version="1.0.0", learning_rate=3e-4, 
                 policy_kwargs=None, verbose=1):
        super().__init__(name)
        self.version = version
        self.env = env
        
        # Enhanced policy network architecture for poker
        if policy_kwargs is None:
            policy_kwargs = {
                'net_arch': [256, 256, 128],  # Deeper network for complex poker decisions
                'activation_fn': torch.nn.ReLU,
            }
        
        # Use MaskablePPO for tournament poker (handles invalid actions)
        self.model = MaskablePPO(
            "MlpPolicy", 
            env,
            learning_rate=learning_rate,
            n_steps=2048,  # Steps per update
            batch_size=64,
            n_epochs=10,
            gamma=0.995,  # High discount for long tournament episodes
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=42
        )
        
        # Training statistics with proper typing (allow str values for metadata)
        self.training_stats: Dict[str, Union[int, float, str]] = {
            'total_timesteps': 0,
            'tournaments_played': 0,
            'average_placement': 0.0,
            'win_rate': 0.0
        }
    
    def act(self, observation, action_mask=None, deterministic=True):
        """Get action from the trained model"""
        try:
            if action_mask is not None:
                action, _ = self.model.predict(observation, action_masks=action_mask, deterministic=deterministic)
            else:
                action, _ = self.model.predict(observation, deterministic=deterministic)
            return int(action)
        except Exception as e:
            print(f"Error in Sharky action prediction: {e}")
            # Fallback to safe action (fold if possible, otherwise check/call)
            if action_mask is not None and len(action_mask) >= 3:
                if action_mask[0]:  # Can fold
                    return 0
                elif action_mask[1]:  # Can call/check
                    return 1
                elif action_mask[2]:  # Can raise
                    return 2
            return 1  # Default to call/check
    
    def learn(self, total_timesteps=50000, callback=None):
        """Train the agent"""
        print(f"🦈 Starting Sharky {self.version} training for {total_timesteps} timesteps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        self.training_stats['total_timesteps'] = int(self.training_stats.get('total_timesteps', 0)) + total_timesteps
        print(f"✅ Sharky {self.version} training completed!")
        
        return self.model
    
    def save(self, path: str):
        """Save the trained model"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        self.model.save(path)
        
        # Save training stats
        stats_path = path.replace('.zip', '_stats.npy')
        np.save(stats_path, self.training_stats, allow_pickle=True)  # type: ignore
        
        print(f"💾 Sharky {self.version} saved to {path}")
    
    def load(self, path: str):
        """Load a pre-trained model"""
        if os.path.exists(path):
            self.model = MaskablePPO.load(path, env=self.env)
            
            # Load training stats if they exist
            stats_path = path.replace('.zip', '_stats.npy')
            if os.path.exists(stats_path):
                self.training_stats = np.load(stats_path, allow_pickle=True).item()
            
            print(f"📂 Sharky {self.version} loaded from {path}")
            return True
        else:
            print(f"❌ Model file not found: {path}")
            return False
    
    def get_name(self):
        """Get agent name with version"""
        return f"{self.name}_{self.version}"
    
    def clone(self, new_version: str):
        """Create a copy of this agent with a new version"""
        new_agent = SharkyAgent(
            env=self.env,
            name=self.name,
            version=new_version,
            verbose=0
        )
        
        # Copy the current model weights
        new_agent.model.set_parameters(self.model.get_parameters())  # type: ignore
        new_agent.training_stats = self.training_stats.copy()
        
        return new_agent


class TournamentCallback(BaseCallback):
    """Callback to log tournament-specific metrics during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.tournament_rewards = []
        self.placements = []
    
    def _on_step(self) -> bool:
        # Log tournament completion
        if self.locals.get('dones', [False])[0]:  # Tournament finished
            if hasattr(self.training_env, 'envs') and hasattr(self.training_env.envs[0], 'elimination_order'):  # type: ignore
                env = self.training_env.envs[0]  # type: ignore
                total_players = env.total_players
                eliminated = len(env.elimination_order)
                
                if eliminated < total_players:
                    placement = 1  # Winner
                else:
                    placement = total_players  # Last place (shouldn't happen)
                
                self.placements.append(placement)
                
                if len(self.placements) % 10 == 0:
                    avg_placement = np.mean(self.placements[-10:])
                    win_rate = len([p for p in self.placements[-10:] if p == 1]) / 10
                    print(f"Recent 10 tournaments - Avg placement: {avg_placement:.1f}, Win rate: {win_rate:.1%}")
        
        return True