# env/poker_env.py

import numpy as np
import random
import gym
from gym import spaces
from engine.game import PokerGame
from engine.player import Player
from utils.enums import GameMode

class PokerEnv(gym.Env):
    """
    A simple Gym-like poker environment for 2 players with discrete actions:
    0 = fold, 1 = call/check, 2 = raise
    """

    def __init__(self):
        super(PokerEnv, self).__init__()
        self.players = [Player("P1", stack=1000), Player("P2", stack=1000)]
        self.game = PokerGame(self.players, game_mode=GameMode.AI_VS_AI)
        
        # Actions: fold, call/check, raise
        self.action_space = spaces.Discrete(3)

        # Observations: dummy 10-element vector for now
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        self.current_player_index = 0
        self.done = False

    def reset(self):
        for p in self.players:
            p.stack = 1000
        self.game = PokerGame(self.players, game_mode=GameMode.AI_VS_AI)
        self.game.play_hand()
        self.done = False
        self.current_player_index = 0
        return self._get_observation()

    def step(self, action):
        """
        Takes a fake action, updates dummy reward logic, and ends episode randomly.
        Real action handling will come later.
        """
        reward = random.choice([-1, 0, 1])  # stub reward logic
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

        self.done = random.random() < 0.2  # 20% chance to end episode
        obs = self._get_observation()
        info = {}
        return obs, reward, self.done, info

    def _get_observation(self):
        # Dummy fixed-size observation for now
        return np.random.rand(10).astype(np.float32)
