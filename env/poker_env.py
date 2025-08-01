import gym
from gym import spaces
import numpy as np
from engine.game import PokerGame
from engine.player import Player

class PokerEnv(gym.Env):
    """
    Gym-compatible Poker environment for RL agents.
    Supports self-play and agent-vs-agent matches.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_players=2, starting_stack=1000, small_blind=10, big_blind=20):
        super().__init__()
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind

        # Discrete action space: 0=fold, 1=call/check, 2=raise
        self.action_space = spaces.Discrete(3)
        # Observation space: [player_stack, to_call, pot, current_bet, is_in_hand]
        self.observation_space = spaces.Box(
            low=0, high=1e6, shape=(5,), dtype=np.float32
        )

        self.players = [Player(f"Agent_{i}", stack=starting_stack) for i in range(num_players)]
        self.game = PokerGame(self.players, small_blind=small_blind, big_blind=big_blind)
        self.current_player_idx = 0

    def legal_action_mask(self):
        idx = self.game.current_player_idx or 0
        player = self.players[idx]
        to_call = self.game.current_bet - player.current_bet
        # [fold, call/check, raise]
        mask = [to_call > 0, True, True]
        return np.array(mask, dtype=bool)

    def reset(self):
        self.players = [Player(f"Agent_{i}", stack=self.starting_stack) for i in range(self.num_players)]
        self.game = PokerGame(self.players, small_blind=self.small_blind, big_blind=self.big_blind)
        self.game.reset_for_new_hand()
        self.current_player_idx = self.game.current_player_idx or 0
        # Track stacks for reward calculation
        self.prev_stacks = {p.name: p.stack for p in self.players}
        obs = self._get_obs()
        return obs

    def step(self, action):
        idx = self.game.current_player_idx or 0
        player = self.players[idx]
        to_call = self.game.current_bet - player.current_bet

        # Map discrete action to poker action
        if action == 0:
            poker_action = "fold"
            raise_amount = 0
        elif action == 1:
            poker_action = "call" if to_call > 0 else "check"
            raise_amount = 0
        elif action == 2:
            poker_action = "raise"
            # For testing, just raise the minimum allowed
            raise_amount = self.game.current_bet + self.game.big_blind
        else:
            raise ValueError(f"Invalid action: {action}")

        self.game.step(poker_action, raise_amount)
        obs = self._get_obs()
        reward = self._get_reward(player)
        done = self.game.hand_over
        info = {"action_mask": self.legal_action_mask()}

        self.prev_stacks[player.name] = player.stack

        return obs, reward, done, info

    def _get_reward(self, player):
        # Reward is change in stack since last step
        prev = self.prev_stacks.get(player.name, self.starting_stack)
        reward = player.stack - prev

        # If hand is over, give a bonus for chip leader
        if self.game.hand_over:
            stacks = [p.stack for p in self.players]
            max_stack = max(stacks)
            if player.stack == max_stack:
                # Chip leader bonus
                reward += 10.0
            # Optionally, add a smaller bonus for 2nd place, etc.
            # You can also encode the player's rank as a reward
        return reward

    def render(self, mode="human"):
        print(f"Pot: {self.game.pot}")
        for p in self.players:
            print(f"{p.name}: stack={p.stack}, bet={p.current_bet}, in_hand={p.in_hand}")

    def _get_obs(self):
        idx = self.game.current_player_idx or 0
        player = self.players[idx]
        obs = np.array([
            player.stack,
            self.game.current_bet - player.current_bet,
            self.game.pot,
            player.current_bet,
            int(player.in_hand)
        ], dtype=np.float32)
        return obs
