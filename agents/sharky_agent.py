from agents.base_rl_agent import BaseRLAgent
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

class SharkyAgent(BaseRLAgent):
    def __init__(self, env, name="Sharky", use_maskable_ppo=False):
        super().__init__(name)
        if use_maskable_ppo:
            self.model = MaskablePPO("MlpPolicy", env, verbose=1)
        else:
            self.model = PPO("MlpPolicy", env, verbose=1)

    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def learn(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)