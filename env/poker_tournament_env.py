import gymnasium as gym
import numpy as np
from typing import List, Optional, Tuple
from engine.player import Player
from engine.game import PokerGame
from engine.action_validation import validate_raise
import builtins
# builtins.print = lambda *args, **kwargs: None

class PokerTournamentEnv(gym.Env):
    """
    Poker tournament environment for RL.
    Plays until one player remains. Blinds increase on a schedule.
    Rewards are based on placement.
    """
    def __init__(self, num_players: int = 9, starting_stack: int = 1000, blinds_schedule: Optional[List[Tuple[int, int, int]]] = None, hands_per_level: int = 25):
        super().__init__()
        self.num_players: int = num_players
        self.starting_stack: int = starting_stack
        self.hands_per_level: int = hands_per_level
        self.blinds_schedule: List[Tuple[int, int, int]] = blinds_schedule or [
            (10, 20, 0),     # Level 1 - no ante
            (15, 30, 0),     # Level 2 - no ante  
            (25, 50, 1),     # Level 3 - antes begin (total ante = BB)
            (50, 100, 1),    # Level 4 - antes continue
            (75, 150, 1),    # Level 5 - antes continue
            (100, 200, 1),   # Level 6 - antes continue
            (150, 300, 1),   # Level 7 - antes continue
            (200, 400, 1),   # Level 8 - antes continue
            (300, 600, 1),   # Level 9 - antes continue
            (500, 1000, 1),  # Level 10 - antes continue
            (750, 1500, 1),  # Level 11 - antes continue
            (1000, 2000, 1), # Level 12 - antes continue
        ]
        self._validate_blind_schedule()
        self.current_blind_level: int = 0
        self.hands_played: int = 0
        self.dealer_position: int = 0  # Track dealer position across resets
        self.players: List[Player] = []
        self.elimination_order: List[Player] = []
        self._setup_players()
        from engine.game import PokerGame
        blind_level = self.blinds_schedule[self.current_blind_level]
        sb, bb, ante = blind_level
        self.game: PokerGame = PokerGame(self.players, starting_stack=self.starting_stack, 
                             small_blind=sb, big_blind=bb, ante=ante)
        self._setup_game()
        self.observation_space = gym.spaces.Box(
            low=0, high=1e6, shape=(5,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # fold, call/check, raise
        self.prev_stacks: dict[str, int] = {}

    def _validate_blind_schedule(self):
        """Validate and normalize blind schedule to enforce consistent ante logic"""
        antes_started = False
        
        for i, level in enumerate(self.blinds_schedule):
            if len(level) == 2:
                # Convert (sb, bb) to (sb, bb, 0)
                self.blinds_schedule[i] = (level[0], level[1], 0)
            elif len(level) == 3:
                sb, bb, ante = level
                
                # Enforce ante consistency rules
                if ante > 0:
                    antes_started = True
                    # Normalize ante to 1 (flag for antes active)
                    self.blinds_schedule[i] = (sb, bb, 1)
                elif antes_started:
                    # Once antes start, they continue for all subsequent levels
                    print(f"Warning: Level {i+1} has ante=0 but antes already started. Setting to 1.")
                    self.blinds_schedule[i] = (sb, bb, 1)
            else:
                raise ValueError(f"Invalid blind level format at level {i+1}: {level}")

    def _setup_players(self):
        self.players = [Player(f"Agent_{i}", stack=self.starting_stack) for i in range(self.num_players)]

    def _setup_game(self):
        blind_level = self.blinds_schedule[self.current_blind_level]
        sb, bb, ante = blind_level
        # Set dealer position before rotating
        self.game.dealer_position = self.dealer_position
        # Rotate dealer before starting a new hand
        self.game.rotate_dealer()
        # Update our tracked dealer position
        self.dealer_position = self.game.dealer_position
        # Debug print for dealer/SB/BB positions
        n = len(self.players)
        print(f"Dealer position: {self.game.dealer_position}, SB: {(self.game.dealer_position + 1) % n}, BB: {(self.game.dealer_position + 2) % n}")
        self.game.small_blind = sb
        self.game.big_blind = bb
        self.game.ante = ante
        self.game.reset_for_new_hand()
        self.prev_stacks = {p.name: p.stack for p in self.players}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # Only reset dealer position on first reset, otherwise preserve rotation
        if not hasattr(self, '_first_reset_done'):
            self.dealer_position = 0
            self._first_reset_done = True
        self.current_blind_level = 0
        self.hands_played = 0
        self.elimination_order = []
        self._setup_players()
        # --- FIX: Re-initialize PokerGame after resetting players ---
        from engine.game import PokerGame
        # Initialize with current blind level for ante support
        blind_level = self.blinds_schedule[self.current_blind_level]
        sb, bb, ante = blind_level
        self.game = PokerGame(self.players, starting_stack=self.starting_stack,
                             small_blind=sb, big_blind=bb, ante=ante)
        self._setup_game()
        self.hand_done = False
        obs = self._get_obs()
        info = {"action_mask": self.legal_action_mask()}
        return obs, info

    def legal_action_mask(self) -> np.ndarray:
        idx = self.game.current_player_idx
        if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(self.players):
            return np.array([False, False, False], dtype=bool)
        player: Player = self.players[idx]
        if player.stack == 0 or not player.in_hand or getattr(player, "all_in", False):
            return np.array([False, False, False], dtype=bool)
        to_call = self.game.current_bet - player.current_bet
        mask = [False, False, False]  # [fold, call/check, raise]
        mask[0] = player.in_hand and to_call > 0
        mask[1] = player.in_hand and player.stack > 0
        min_raise = max(self.game.current_bet + self.game.last_raise_amount, self.game.big_blind)
        max_raise = player.stack + player.current_bet
        mask[2] = (
            player.in_hand and
            player.stack > to_call and
            (max_raise >= min_raise)
        )
        return np.array(mask, dtype=bool)

    def step(self, action: int):
        idx = self.game.current_player_idx
        if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(self.players):
            raise Exception("Invalid current_player_idx; cannot step.")
        player: Player = self.players[idx]
        mask = self.legal_action_mask()
        if not any(mask):
            raise Exception(f"No legal actions remain for player {player.name} (stack={player.stack}, in_hand={player.in_hand}, all_in={getattr(player, 'all_in', False)})")
        if not mask[action]:
            raise Exception(f"Illegal action {action} for player {player.name} (stack={player.stack}, in_hand={player.in_hand})")
        to_call = self.game.current_bet - player.current_bet
        if player.stack == 0:
            if to_call == 0:
                poker_action = "check"
                raise_amount = 0
            else:
                poker_action = "fold"
                raise_amount = 0
        else:
            if to_call == 0 and action == 0:
                poker_action = "check"
                raise_amount = 0
            elif action == 0:
                poker_action = "fold"
                raise_amount = 0
            elif action == 1:
                poker_action = "call" if to_call > 0 else "check"
                raise_amount = 0
            elif action == 2:
                min_raise = max(self.game.current_bet + self.game.last_raise_amount, self.game.big_blind)
                max_raise = player.stack + player.current_bet
                try:
                    can_raise = validate_raise(
                        player_stack=player.stack,
                        to_call=to_call,
                        current_bet=self.game.current_bet,
                        min_raise=self.game.last_raise_amount,
                        big_blind=self.game.big_blind,
                        player_current_bet=player.current_bet,
                        raise_to=min_raise
                    )
                except Exception:
                    can_raise = False
                if can_raise:
                    poker_action = "raise"
                    raise_amount = min(min_raise, max_raise) if max_raise >= min_raise else max_raise
                else:
                    poker_action = "call" if to_call > 0 else "check"
                    raise_amount = 0
            else:
                raise ValueError("Invalid action")
        self.game.step(poker_action, raise_amount)
        reward = self._get_reward(player)
        self.prev_stacks[player.name] = player.stack
        terminated = False
        truncated = False
        info = {"action_mask": self.legal_action_mask()}
        if self.game.hand_over:
            eliminated = [p for p in self.players if p.stack == 0 and p not in self.elimination_order]
            for p in eliminated:
                self.elimination_order.append(p)
                print(f"[ELIMINATED] {p.name} is out! Place: {len(self.players) - len(self.elimination_order) + 1}")
            active_players = [p for p in self.players if p.stack > 0]
            if len(active_players) <= 1:
                terminated = True
                print("[DEBUG] Episode terminated: only one player remains.")
            else:
                self.hands_played += 1
                if self.hands_played % self.hands_per_level == 0 and self.current_blind_level < len(self.blinds_schedule) - 1:
                    self.current_blind_level += 1
                    top_players = sorted(self.players, key=lambda p: p.stack, reverse=True)[:3]
                    print(f"\n[BLINDS INCREASED] New blinds: {self.blinds_schedule[self.current_blind_level]}")
                    for i, p in enumerate(top_players, 1):
                        print(f"Top {i}: {p.name} - Stack: {p.stack}")
                self._setup_game()
                truncated = True
                print("[DEBUG] Episode truncated: game reset for next hand.")
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        idx = self.game.current_player_idx
        if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(self.players):
            return np.zeros(5, dtype=np.float32)
        player: Player = self.players[idx]
        obs = np.array([
            player.stack,
            self.game.current_bet - player.current_bet,
            self.game.pot,
            player.current_bet,
            int(player.in_hand)
        ], dtype=np.float32)
        return obs

    def _get_reward(self, player):
        # Reward is change in stack since last step
        prev = self.prev_stacks.get(player.name, self.starting_stack)
        reward = player.stack - prev
        return reward

    def _get_placement_rewards(self, player):
        # Placement-based reward: only at the end
        placement = {p.name: i+1 for i, p in enumerate(self.elimination_order)}
        for p in self.players:
            if p.stack > 0:
                placement[p.name] = 1  # Winner
        placement_rewards = [100, 50, 30, 20, 10, 5, 3, 2, 1]
        place = placement.get(player.name, len(self.players))
        return placement_rewards[place-1] if place <= len(placement_rewards) else 0

    def render(self, mode="human"):
        print(f"Pot: {self.game.pot}")
        for p in self.players:
            print(f"{p.name}: stack={p.stack}, bet={p.current_bet}, in_hand={p.in_hand}")

    @property
    def current_player_idx(self):
        return self.game.current_player_idx

    def get_obs_for_player(self, player: Player) -> np.ndarray:
        obs = np.array([
            player.stack,
            self.game.current_bet - player.current_bet,
            self.game.pot,
            player.current_bet,
            int(player.in_hand)
        ], dtype=np.float32)
        return obs