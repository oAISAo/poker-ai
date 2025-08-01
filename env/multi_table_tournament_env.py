import gymnasium as gym
import numpy as np
from typing import Dict, List, Optional, Tuple
from engine.player import Player
from engine.game import PokerGame
from engine.action_validation import validate_raise
import builtins
import random
import math

# Suppress print statements for cleaner output
# builtins.print = lambda *args, **kwargs: None  # TEMPORARILY DISABLED FOR DEBUGGING

class Table:
    """Represents a single poker table in the tournament"""
    def __init__(self, table_id: int, players: List[Player], starting_stack: int, 
                 small_blind: int = 10, big_blind: int = 20, ante: int = 0):
        self.table_id = table_id
        self.players = players
        self.game = PokerGame(players, starting_stack=starting_stack, 
                            small_blind=small_blind, big_blind=big_blind, ante=ante)
        self.hands_played = 0
        self.is_active = len(players) >= 2
        self._last_elimination_signature: Optional[Tuple[str, ...]] = None
        
    def get_active_player_count(self) -> int:
        """Get number of players with chips remaining"""
        return len([p for p in self.players if p.stack > 0])
    
    def remove_player(self, player: Player) -> bool:
        """Remove a player from the table. Returns True if successful."""
        if player in self.players:
            self.players.remove(player)
            # Update the game's player list
            self.game.players = self.players
            self.is_active = len(self.players) >= 2
            return True
        return False
    
    def add_player(self, player: Player, seat_position: Optional[int] = None, emergency: bool = False) -> bool:
        """Add a player to the table at specified seat or next available."""
        if not emergency and len(self.players) >= 9:  # Max 9 players per table
            return False
        
        if seat_position is not None and seat_position < len(self.players):
            self.players.insert(seat_position, player)
        else:
            self.players.append(player)
        
        # Update the game's player list
        self.game.players = self.players
        self.is_active = len(self.players) >= 2
        return True

class MultiTableTournamentEnv(gym.Env):
    """
    Multi-table poker tournament environment for RL.
    Supports multiple tables, table balancing, blind increases, and player elimination tracking.
    """
    
    def __init__(self, 
                 total_players: int = 99, 
                 max_players_per_table: int = 9,
                 min_players_per_table: int = 2,
                 starting_stack: int = 1000,
                 blinds_schedule: Optional[List[Tuple[int, int, int]]] = None,
                 hands_per_blind_level: int = 9,
                 table_balancing_threshold: int = 5):
        """
        Initialize multi-table tournament environment.
        
        Args:
            total_players: Total number of players in tournament
            max_players_per_table: Maximum players per table
            min_players_per_table: Minimum players before table breaking (2 for heads-up)
            starting_stack: Starting chip stack for each player
            blinds_schedule: List of (small_blind, big_blind) tuples
            hands_per_blind_level: Hands before blind level increases (9 for turbo)
            table_balancing_threshold: Trigger table balancing when table drops below this (5)
        """
        super().__init__()
        
        # Input validation
        if total_players < 2:
            raise ValueError("Tournament must have at least 2 players")
        if max_players_per_table < 2:
            raise ValueError("Maximum players per table must be at least 2")
        if starting_stack <= 0:
            raise ValueError("Starting stack must be positive")
        if hands_per_blind_level <= 0:
            raise ValueError("Hands per blind level must be positive")
        if table_balancing_threshold < 2:
            raise ValueError("Table balancing threshold must be at least 2")
        
        # Tournament configuration
        self.total_players = total_players
        self.max_players_per_table = max_players_per_table
        self.min_players_per_table = min_players_per_table
        self.starting_stack = starting_stack
        self.table_balancing_threshold = table_balancing_threshold
        self.hands_per_blind_level = hands_per_blind_level
        
        # Realistic turbo tournament blind structure (9-hand levels)
        self.blinds_schedule = blinds_schedule or [
            (10, 20, 0),     # Level 1 - no ante
            (20, 40, 0),     # Level 2 - no ante
            (30, 60, 1),     # Level 3 - antes begin
            (40, 80, 1),     # Level 4 - antes continue
            (50, 100, 1),    # Level 5 - antes continue
            (60, 120, 1),    # Level 6 - antes continue
            (80, 160, 1),    # Level 7 - antes continue
            (100, 200, 1),   # Level 8 - antes continue
            (150, 300, 1),   # Level 9 - antes continue
            (250, 500, 1),   # Level 10 - aggressive escalation for late game
            (400, 800, 1),   # Level 11 - very aggressive
            (600, 1200, 1),  # Level 12 - force action
            (1000, 2000, 1), # Level 13 - extreme pressure
            (1500, 3000, 1), # Level 14 - tournament should end soon
            (2500, 5000, 1), # Level 15 - endgame
            (4000, 8000, 1), # Level 16 - force heads-up conclusion
            (6000, 12000, 1), # Level 17 - emergency level
            (10000, 20000, 1), # Level 18 - final emergency level
        ]
        
        # Validate and normalize blind schedule
        self._validate_blind_schedule()
        
        # Validate blind schedule (now all levels are 3-tuples after normalization)
        if not self.blinds_schedule or len(self.blinds_schedule) == 0:
            raise ValueError("Blind schedule cannot be empty")
        for i, blind_level in enumerate(self.blinds_schedule):
            sb, bb, ante = blind_level  # All levels are 3-tuples after normalization
            
            if sb <= 0 or bb <= 0:
                raise ValueError(f"Invalid blinds at level {i}: ({sb}, {bb})")
            if bb <= sb:
                raise ValueError(f"Big blind must be greater than small blind at level {i}: ({sb}, {bb})")
            if ante < 0:
                raise ValueError(f"Ante cannot be negative at level {i}: {ante}")
        
        # Tournament state
        self.current_blind_level = 0
        self.hands_played_this_level = 0
        self.total_hands_played = 0
        self.tables: Dict[int, Table] = {}
        self.all_players: List[Player] = []
        self.elimination_order: List[Player] = []
        self.active_table_id: int = 0  # Current table being played
        
        # Calculate initial number of tables needed
        self.num_tables = math.ceil(total_players / max_players_per_table)
        
        # Initialize players and tables
        self._setup_tournament()
        
        # Gym spaces - observation for current active player
        self.observation_space = gym.spaces.Box(
            low=0, high=1e8, shape=(8,), dtype=np.float32
        )  # [stack, to_call, pot, current_bet, in_hand, table_id, players_at_table, blind_level]
        
        self.action_space = gym.spaces.Discrete(3)  # fold, call/check, raise
        
        # Track previous stacks for reward calculation
        self.prev_stacks: Dict[str, int] = {}
    
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
    
    def _setup_tournament(self):
        """Initialize all players and distribute them across tables"""
        # Create all players - can be overridden by subclasses for rule-based opponents
        self.all_players = self._create_players()
        
        # Shuffle players for random seating
        random.shuffle(self.all_players)
        
        # Distribute players across tables
        self._distribute_players_to_tables()
        
        # Initialize tracking
        self.prev_stacks = {p.name: p.stack for p in self.all_players}
        print(f"Tournament initialized: {len(self.tables)} tables, {self.total_players} players")
    
    def _create_players(self):
        """Create players - can be overridden for different opponent types"""
        return [
            Player(f"Player_{i}", stack=self.starting_stack) 
            for i in range(self.total_players)
        ]
    
    def _distribute_players_to_tables(self):
        """Distribute players across tables"""
        player_idx = 0
        for table_id in range(self.num_tables):
            # Calculate players for this table
            remaining_players = self.total_players - player_idx
            remaining_tables = self.num_tables - table_id
            players_for_table = min(
                self.max_players_per_table,
                math.ceil(remaining_players / remaining_tables)
            )
            
            # Assign players to this table
            table_players = self.all_players[player_idx:player_idx + players_for_table]
            player_idx += players_for_table
            
            # Create table with current blind level
            blind_level = self.blinds_schedule[self.current_blind_level]
            sb, bb, ante = blind_level  # All levels are 3-tuples after normalization
            table = Table(table_id, table_players, self.starting_stack, sb, bb, ante)
            self.tables[table_id] = table
            
            # Initialize the table's game (first hand)
            table.game.reset_for_new_hand(is_first_hand=True)
    
    def _get_active_tables(self) -> List[Table]:
        """Get all tables that are still active (have 2+ players)"""
        return [table for table in self.tables.values() if table.is_active]
    
    def _select_next_active_table(self) -> Optional[int]:
        """Select the next table that needs to play a hand"""
        active_tables = self._get_active_tables()
        if not active_tables:
            return None
        
        # Round-robin table selection
        current_active_table_ids = [t.table_id for t in active_tables]
        current_active_table_ids.sort()
        
        # Find next table after current active table
        try:
            # Type guard: ensure active_table_id is not None before using index
            if self.active_table_id is not None:
                current_idx = current_active_table_ids.index(self.active_table_id)
                next_idx = (current_idx + 1) % len(current_active_table_ids)
                return current_active_table_ids[next_idx]
            else:
                # active_table_id is None, start from first
                return current_active_table_ids[0]
        except ValueError:
            # Current table not in active list, start from first
            return current_active_table_ids[0]
    
    def balance_table(self, table_id):
        """
        Rebalance a single table after its hand is finished.
        If table needs players, move them from other tables with surplus.
        If table has surplus, move players to tables needing them.
        Handles multiple players needed/given.
        """

        print(f"[DEBUG] Entered balance_table for table_id: {table_id}")
        table = self.tables.get(table_id)
        # Never forcibly end a hand due to eliminations; wait for hand to finish naturally
        # If table is in a hand, skip balancing until hand is over
        if not table or not table.is_active:
            print(f"[DEBUG] Table {table_id} is not active; skipping balancing.")
            return
        if not table.game.hand_over:
            print(f"[DEBUG] Table {table_id} is still in a hand; skipping balancing")
            return

        current = table.get_active_player_count()
        if current < self.min_players_per_table:
            print(f"[DEBUG] Table {table_id} breaking: only {current} active players.")
            self._break_table(table)
            print(f"[DEBUG] Table {table_id} deactivated after breaking.")
            return

        active_tables = [t for t in self.tables.values() if t.is_active and t.get_active_player_count() > 0]
        if len(active_tables) <= 1:
            print(f"[DEBUG] Only one active table; no balancing needed.")
            return

        # Calculate ideal player count per table
        total_players = sum(t.get_active_player_count() for t in active_tables)
        num_tables = len(active_tables)
        ideal = total_players // num_tables
        extra = total_players % num_tables

        # For each table, calculate target count
        targets = [ideal + (1 if i < extra else 0) for i in range(num_tables)]
        table_map = {t.table_id: t for t in active_tables}
        table_ids_sorted = sorted(table_map.keys())
        target_map = {tid: targets[i] for i, tid in enumerate(table_ids_sorted)}

        # How many players does this table need/give?
        target = target_map[table_id]
        if current == target:
            print(f"[DEBUG] Table {table_id} already balanced ({current} players).")
            return
        elif current < target:
            # If table is below min_players_per_table, break and deactivate it
            print(f"[DEBUG] Table {table_id} is below min_players_per_table after breaking; breaking and deactivating.")
            self._break_table(table)
            print(f"[DEBUG] Table {table_id} deactivated after breaking (balancing phase).")
            # Ensure at least one table is active and has a valid hand started
            active_tables = [t for t in self.tables.values() if t.is_active and t.get_active_player_count() >= 2]
            if active_tables:
                # Set active_table_id to a playable table
                self.active_table_id = active_tables[0].table_id
                # If the table's hand is over, reset for new hand
                if active_tables[0].game.hand_over:
                    try:
                        blind_level = self.blinds_schedule[self.current_blind_level]
                        sb, bb, ante = blind_level
                        active_tables[0].game.small_blind = sb
                        active_tables[0].game.big_blind = bb
                        active_tables[0].game.ante = ante
                        active_tables[0].game.reset_for_new_hand(is_first_hand=False)
                        print(f"[DEBUG] Started new hand at table {active_tables[0].table_id} after breaking.")
                    except Exception as e:
                        print(f"[DEBUG] Error resetting hand after breaking: {e}")
            return
        else:
            players_to_give = current - target
            print(f"[DEBUG] Table {table_id} has {players_to_give} surplus player(s) (target {target}).")
            # Find tables needing players
            for receiver_id in table_ids_sorted:
                if receiver_id == table_id:
                    continue
                receiver_table = table_map[receiver_id]
                receiver_current = receiver_table.get_active_player_count()
                receiver_target = target_map[receiver_id]
                need = receiver_target - receiver_current
                while need > 0 and players_to_give > 0:
                    moveable = [p for p in table.players if p.stack > 0]
                    if not moveable:
                        break
                    player_to_move = moveable[0]
                    table.remove_player(player_to_move)
                    receiver_table.add_player(player_to_move)
                    # Player moved mid-hand joins next hand only
                    if receiver_table.game.hand_over:
                        player_to_move.in_hand = True
                    else:
                        player_to_move.in_hand = False
                    print(f"[DEBUG] Moved {player_to_move.name} from table {table_id} to table {receiver_id}")
                    need -= 1
                    players_to_give -= 1
        # Fix game state for all affected tables
        for t in active_tables:
            self._fix_game_state_after_eliminations(t)
        print(f"[DEBUG] Table {table_id} balancing complete.")
    
    def _fix_game_state_after_eliminations(self, table: Table):
        """Fix game state after manual eliminations (e.g., setting stack=0)"""
        # Only fix if there are actually eliminated players
        eliminated_players = [p for p in table.players if p.stack == 0]
        if not eliminated_players:
            return  # No eliminations, no fix needed
        
        # Create a signature for this elimination state to avoid duplicate messages
        elimination_signature = tuple(sorted(p.name for p in eliminated_players))
        last_signature = getattr(table, '_last_elimination_signature', None)
        
        # Remove eliminated players from the hand (double-check they actually have 0 stack)
        for player in eliminated_players:
            if player.stack == 0:  # Safety check to prevent zombie players
                player.in_hand = False
        
        # CRITICAL FIX: Restore players with chips to active status if incorrectly marked as inactive
        for player in table.players:
            if player.stack > 0 and not player.in_hand:
                player.in_hand = True
                print(f"[DEBUG] Restored {player.name} to active status (stack: {player.stack})")
                
                # CRITICAL FIX: Remove from elimination_order if player was incorrectly eliminated
                if player in self.elimination_order:
                    self.elimination_order.remove(player)
                    print(f"[DEBUG] Removed {player.name} from elimination_order (stack restored: {player.stack})")
        
        # Find active players
        active_players = [p for p in table.players if p.stack > 0]

        # If hand is in progress and only one player remains in_hand with chips, end the hand immediately (real poker rule)
        if not table.game.hand_over:
            in_hand_players = [p for p in table.players if p.stack > 0 and p.in_hand]
            if len(in_hand_players) == 1:
                table.game.hand_over = True
                print(f"[DEBUG] Hand ended early at table {table.table_id} due to only one player remaining after eliminations ({in_hand_players[0].name})")
                # Optionally, award the pot to the last player (if not already handled by PokerGame)
                # You may want to call a PokerGame method here if needed

        if len(active_players) >= 2:
            # Reset the game with remaining players if we have enough for a game
            try:
                # Set current player to first active player
                active_indices = [i for i, p in enumerate(table.players) if p.stack > 0]
                if active_indices:
                    table.game.current_player_idx = active_indices[0]
                    # Only reset game state if hand is over
                    if table.game.hand_over:
                        table.game.hand_over = False
                        for player in active_players:
                            player.in_hand = True
                            player.current_bet = 0
                        table.game.pot = 0
                        table.game.current_bet = 0
                        table.game.community_cards = []
                        table.game.phase_idx = 0
                        # Only print if this is a new elimination state
                        if elimination_signature != last_signature:
                            print(f"[DEBUG] Fixed game state for table {table.table_id} with {len(active_players)} active players")
                            table._last_elimination_signature = elimination_signature
                    else:
                        # Hand is in progress, don't reset pot/bets, just remove eliminated players
                        # Only print if this is a new elimination state
                        if elimination_signature != last_signature:
                            print(f"[DEBUG] Removed {len(eliminated_players)} eliminated players from table {table.table_id}")
                            table._last_elimination_signature = elimination_signature
            except Exception as e:
                print(f"Error fixing game state for table {table.table_id}: {e}")
                table.is_active = False
    
    def _break_table(self, table: Table):
        """Break a table and distribute its players to other tables"""
        active_players = [p for p in table.players if p.stack > 0]
        
        # Find tables that can accept players (exclude the table being broken)
        potential_tables = [t for t in self.tables.values() if t.table_id != table.table_id]
        # Include any table that has players with chips (not just those with >=2 players)
        available_tables = [t for t in potential_tables if t.get_active_player_count() > 0]
        
        if not available_tables:
            # No other tables to move players to
            table.is_active = False
            print(f"Table {table.table_id} broken - no other available tables")
            return
        
        for player in active_players:
            # Find table with fewest players (allow going up to max+1 in emergency situations)
            target_table = min(available_tables, key=lambda t: len(t.players))
            # Allow moving even if table is at max capacity (emergency table breaking)
            table.remove_player(player)
            target_table.add_player(player, emergency=True)
            print(f"Player {player.name} moved from table {table.table_id} to {target_table.table_id}")
        
        # Deactivate the broken table
        table.is_active = False
        print(f"Table {table.table_id} broken")
    
    def _move_player_between_tables(self, source_table: Table, target_table: Table):
        """Move one player from source table to target table"""
        # Find a player not currently in a hand to move
        moveable_players = [p for p in source_table.players if p.stack > 0]
        
        if moveable_players and len(target_table.players) < self.max_players_per_table:
            player_to_move = random.choice(moveable_players)
            source_table.remove_player(player_to_move)
            target_table.add_player(player_to_move)
            print(f"Player {player_to_move.name} moved from table {source_table.table_id} to {target_table.table_id}")
    
    def _increase_blinds_if_needed(self):
        """Check if blinds should increase and apply to all tables"""
        if self.hands_played_this_level >= self.hands_per_blind_level:
            if self.current_blind_level < len(self.blinds_schedule) - 1:
                self.current_blind_level += 1
                self.hands_played_this_level = 0
                
                # Apply new blinds and antes to all active tables
                blind_level = self.blinds_schedule[self.current_blind_level]
                sb, bb, ante = blind_level  # All levels are 3-tuples after normalization
                    
                for table in self._get_active_tables():
                    table.game.small_blind = sb
                    table.game.big_blind = bb
                    table.game.ante = ante
                
                print(f"Blinds increased to {sb}/{bb} (Level {self.current_blind_level + 1})")
            else:
                # At maximum blind level, reset counter to prevent overflow
                self.hands_played_this_level = 0
    
    def _get_ordinal(self, n):
        """Convert number to ordinal (1st, 2nd, 3rd, etc.)"""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
    
    def _clean_elimination_order(self):
        """Remove players from elimination_order if they have chips (shouldn't be eliminated)"""
        players_to_remove = []
        for player in self.elimination_order:
            if player.stack > 0:
                players_to_remove.append(player)
                print(f"[DEBUG] Removing {player.name} from elimination_order (stack: {player.stack})")
        
        for player in players_to_remove:
            self.elimination_order.remove(player)
    
    def _update_elimination_order(self):
        """Update elimination order with players who just busted"""
        for table in self.tables.values():
            for player in table.players:
                if player.stack == 0 and player not in self.elimination_order:
                    # Mark eliminated player as out of hand immediately
                    player.in_hand = False
                    
                    self.elimination_order.append(player)
                    elimination_position = len(self.elimination_order)
                    final_placement = self.total_players - elimination_position + 1
                    print(f"Player {player.name} eliminated ({self._get_ordinal(elimination_position)} elimination, finishes {self._get_ordinal(final_placement)} place)")
                    
                    # Show Sharky's stack after each elimination - find current stack from all tables
                    sharky_player = None
                    for t in self.tables.values():
                        for p in t.players:
                            if p.name == "Player_0" and p.stack > 0:
                                sharky_player = p
                                break
                        if sharky_player:
                            break
                    
                    if sharky_player:
                        print(f"🦈 Sharky (Player_0) stack: {sharky_player.stack} chips")
    
    def _tournament_finished(self) -> bool:
        """Check if tournament is finished (2 or fewer players remain - heads-up should be tested separately)"""
        remaining_players = [p for p in self.all_players if p.stack > 0]
        return len(remaining_players) <= 2
    
    def _calculate_reward(self, player, prev_stack):
        """Calculate comprehensive tournament reward"""
        # 1. Immediate stack change (small weight for short-term feedback)
        stack_change = (player.stack - prev_stack) * 0.1
        
        # 2. Placement reward (only when eliminated or tournament ends)
        placement_reward = 0
        if player.stack == 0 and prev_stack > 0:  # Just eliminated
            elimination_position = len([p for p in self.all_players if p.stack == 0])  # Current elimination position
            final_placement = self.total_players - elimination_position + 1  # Convert to final placement
            placement_reward = self._get_placement_reward(final_placement)
            print(f"[DEBUG] {player.name} eliminated ({self._get_ordinal(elimination_position)} elimination, finishes {self._get_ordinal(final_placement)} place), placement reward: {placement_reward}")
        elif self._tournament_finished() and player.stack > 0:  # Winner
            placement_reward = self._get_placement_reward(1)  # Winner reward
            print(f"[DEBUG] {player.name} wins tournament! Final placement: 1st, placement reward: {placement_reward}")
        
        # 3. Survival bonus (small bonus for surviving each action while in hand)
        survival_bonus = 0
        if prev_stack > 0 and player.stack > 0 and player.in_hand:
            survival_bonus = 0.5  # Small bonus for surviving each action
        
        # 4. Blind level progression bonus (reward for surviving blind increases)
        progression_bonus = 0
        if self.current_blind_level > 0 and player.stack > 0:
            progression_bonus = self.current_blind_level * 2  # Bonus increases with blind level survived
        
        total_reward = stack_change + placement_reward + survival_bonus + progression_bonus
        
        return total_reward
    
    def _get_placement_reward(self, placement):
        """Placement-based rewards for 18-player tournament (scales with tournament size)"""
        # Dynamic rewards based on tournament size
        max_players = self.total_players
        
        if max_players <= 9:  # Single table
            rewards = [500, 300, 200, 150, 100, 75, 50, 25, 10]
        elif max_players <= 18:  # 2 tables
            rewards = [1000, 600, 400, 300, 200, 150, 100, 75, 50, 
                      40, 30, 25, 20, 15, 12, 10, 8, 6]
        elif max_players <= 27:  # 3 tables
            rewards = [1500, 900, 600, 450, 300, 225, 150, 110, 75, 
                      60, 45, 38, 30, 25, 20, 18, 15, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1]
        else:  # Large tournaments
            # Generate rewards for large tournaments
            rewards = []
            for i in range(max_players):
                if i == 0:
                    rewards.append(2000)  # Winner
                elif i < 3:
                    rewards.append(1200 - i * 300)  # Top 3
                elif i < 9:
                    rewards.append(600 - (i - 3) * 50)  # Final table
                elif i < max_players // 2:
                    rewards.append(max(50, 300 - (i - 9) * 10))  # ITM (in the money)
                else:
                    rewards.append(max(1, 50 - (i - max_players // 2)))  # Consolation
        
        return rewards[placement - 1] if placement <= len(rewards) else 0
    
    def reset(self, *, seed=None, options=None):
        """Reset the tournament to initial state"""
        super().reset(seed=seed, options=options)
        
        # Seed the random number generator if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset tournament state
        self.current_blind_level = 0
        self.hands_played_this_level = 0
        self.total_hands_played = 0
        self.elimination_order = []
        self.active_table_id = 0
        
        # Clear existing tables
        self.tables.clear()
        
        # Reinitialize tournament
        self._setup_tournament()
        
        # Select first active table
        self.active_table_id = self._select_next_active_table() or 0
        
        obs = self._get_obs()
        info = {"action_mask": self.legal_action_mask()}
        return obs, info
    
    def _get_obs(self) -> np.ndarray:
        """Get observation for current active player"""
        if self.active_table_id not in self.tables:
            # No active table, return zero observation
            return np.zeros(8, dtype=np.float32)
        
        table = self.tables[self.active_table_id]
        if not table.players or table.game.current_player_idx is None or table.game.current_player_idx >= len(table.players):
            return np.zeros(8, dtype=np.float32)
        
        current_player = table.players[table.game.current_player_idx]
        
        # Ensure all values are valid
        stack = max(0, current_player.stack)
        to_call = max(0, table.game.current_bet - current_player.current_bet)
        pot = max(0, table.game.pot)
        current_bet = max(0, current_player.current_bet)
        in_hand = 1 if current_player.in_hand else 0
        table_id = max(0, table.table_id)
        players_at_table = max(0, len(table.players))
        blind_level = max(0, self.current_blind_level)
        
        obs = np.array([
            stack, to_call, pot, current_bet, in_hand, 
            table_id, players_at_table, blind_level
        ], dtype=np.float32)
        
        # Sanity check - replace any invalid values with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs
    
    def legal_action_mask(self) -> np.ndarray:
        """Generate legal action mask for current player"""
        if self.active_table_id not in self.tables:
            return np.array([False, False, False], dtype=bool)
        
        table = self.tables[self.active_table_id]
        if not table.players or table.game.current_player_idx is None or table.game.current_player_idx >= len(table.players):
            return np.array([False, False, False], dtype=bool)
        
        player = table.players[table.game.current_player_idx]
        
        # If player is all-in or eliminated, no legal actions
        if player.stack == 0 or not player.in_hand or getattr(player, "all_in", False):
            return np.array([False, False, False], dtype=bool)
        
        to_call = table.game.current_bet - player.current_bet
        mask = [False, False, False]  # [fold, call/check, raise]
        
        # Fold: only legal if player is in hand and to_call > 0
        mask[0] = player.in_hand and to_call > 0
        
        # Call/Check: legal if player is in hand and has chips
        # Player can call even if they don't have enough chips (all-in)
        mask[1] = player.in_hand and player.stack > 0
        
        # Raise: legal if player can raise minimum amount or go all-in above current bet
        min_raise_to = max(table.game.current_bet + table.game.last_raise_amount, table.game.big_blind)
        max_possible_raise = player.stack + player.current_bet
        mask[2] = (
            player.in_hand and
            player.stack > to_call and
            max_possible_raise > max(min_raise_to, player.current_bet + 1)  # Must be above current bet
        )
        
        return np.array(mask, dtype=bool)
    
    def step(self, action: int):
        """Execute one step in the tournament"""
        # Validate action input
        if not isinstance(action, (int, np.integer)):
            action = int(action)  # Try to convert
        
        if action < 0 or action >= 3:  # self.action_space.n = 3 (fold, call, raise)
            # Invalid action - return penalty and continue
            obs = self._get_obs()
            return obs, -10, False, False, {"action_mask": self.legal_action_mask()}
        
        if self.active_table_id not in self.tables:
            # No active table, tournament might be finished
            obs = self._get_obs()
            return obs, 0, True, False, {"action_mask": self.legal_action_mask()}
        
        table = self.tables[self.active_table_id]
        
        # Validate state consistency at start of step
        if hasattr(table.game, '_validate_state_consistency'):
            if not table.game._validate_state_consistency(f"start of tournament step - action {action}"):
                print(f"[WARNING] Fixing state inconsistency at start of tournament step")
                table.game.fix_state_inconsistencies()
        
        if not table.players:
            # No players at table, move to next
            next_table = self._select_next_active_table()
            self.active_table_id = next_table if next_table is not None else 0
            obs = self._get_obs()
            return obs, 0, False, False, {"action_mask": self.legal_action_mask()}
        
        # Check if current player index is valid
        if table.game.current_player_idx is None or table.game.current_player_idx >= len(table.players):
            # Invalid player index, try to fix or move to next table
            if table.players:
                table.game.current_player_idx = 0
            else:
                next_table = self._select_next_active_table()
                self.active_table_id = next_table if next_table is not None else 0
                obs = self._get_obs()
                return obs, 0, False, False, {"action_mask": self.legal_action_mask()}
        
        player = table.players[table.game.current_player_idx]
        
        # Validate action against mask
        mask = self.legal_action_mask()
        if not any(mask):
            # No legal actions for this player - skip to next player
            print(f"[DEBUG] Player {player.name} has no legal actions (in_hand={player.in_hand}, stack={player.stack}), advancing to next player")
            
            # Try to advance to next player in the game engine
            try:
                if hasattr(table.game, '_advance_to_next_player'):
                    table.game._advance_to_next_player()
                else:
                    # Fallback: manually advance to next active player
                    active_players = [p for p in table.players if p.stack > 0 and p.in_hand]
                    if not active_players:
                        # No active players in current hand - check if hand should end
                        if hasattr(table.game, 'hand_over') and not table.game.hand_over:
                            # Force hand to end and start a new one
                            table.game.hand_over = True
                            print(f"[DEBUG] Forcing hand to end - no active players")
                        obs = self._get_obs()
                        return obs, 0, False, False, {"action_mask": self.legal_action_mask()}
                    
                    # Find next active player
                    current_idx = table.game.current_player_idx
                    if current_idx is not None:  # Type guard for current_idx
                        for i in range(1, len(table.players)):
                            next_idx = (current_idx + i) % len(table.players)
                            next_player = table.players[next_idx]
                            if next_player.stack > 0 and next_player.in_hand:
                                table.game.current_player_idx = next_idx
                                break
            except Exception as e:
                print(f"[DEBUG] Error advancing player: {e}")
            
            obs = self._get_obs()
            return obs, 0, False, False, {"action_mask": self.legal_action_mask()}
        
        if not mask[action]:
            # Illegal action, return penalty
            obs = self._get_obs()
            return obs, -5, False, False, {"action_mask": self.legal_action_mask()}
        
        # Convert action to poker action
        to_call = max(0, table.game.current_bet - player.current_bet)
        
        if player.stack == 0:
            poker_action = "check" if to_call == 0 else "fold"
            raise_amount = 0
        else:
            if action == 0:  # Fold
                poker_action = "fold"
                raise_amount = 0
            elif action == 1:  # Call/Check
                poker_action = "call" if to_call > 0 else "check"
                raise_amount = 0
            elif action == 2:  # Raise
                # Debug game state before processing raise
                print(f"[DEBUG] Raise attempt: {player.name}, player.current_bet={player.current_bet}, game.current_bet={table.game.current_bet}, to_call={to_call}")
                
                # Check for inconsistent state and fix it using the game's validation system
                if not table.game._validate_state_consistency(f"before raise by {player.name}"):
                    print(f"[WARNING] State inconsistency detected before raise - attempting fix...")
                    table.game.fix_state_inconsistencies()
                    # Recalculate to_call after fixing state
                    to_call = max(0, table.game.current_bet - player.current_bet)
                    
                    # Recalculate to_call after potential fix
                    to_call = table.game.current_bet - player.current_bet
                    
                    # If still inconsistent, fall back to safe action
                    if player.current_bet > table.game.current_bet:
                        print(f"[WARNING] Could not fix inconsistency, forcing safe action")
                        poker_action = "check" if to_call <= 0 else "fold"
                        raise_amount = 0
                    else:
                        print(f"[DEBUG] State inconsistency resolved, proceeding with raise logic")
                        # FIXED: Correct minimum raise calculation
                        # Minimum raise = current_bet + max(last_raise_amount, big_blind)
                        min_raise_increment = max(table.game.last_raise_amount, table.game.big_blind)
                        min_raise_to = table.game.current_bet + min_raise_increment
                        max_possible = player.stack + player.current_bet
                        
                        print(f"[DEBUG] Raise calculation: current_bet={table.game.current_bet}, min_raise_increment={min_raise_increment}, min_raise_to={min_raise_to}, max_possible={max_possible}")
                        
                        if max_possible >= min_raise_to:
                            # Can make a legal raise
                            raise_amount = min_raise_to
                            poker_action = "raise"
                        elif max_possible > table.game.current_bet:
                            # Can't make min raise but can go all-in
                            raise_amount = max_possible
                            poker_action = "raise"  # All-in raise
                        else:
                            # Can't raise, must call or fold
                            poker_action = "call" if to_call > 0 and player.stack >= to_call else "fold"
                            raise_amount = 0
                else:
                    # FIXED: Correct minimum raise calculation (same as above)
                    min_raise_increment = max(table.game.last_raise_amount, table.game.big_blind)
                    min_raise_to = table.game.current_bet + min_raise_increment
                    max_possible = player.stack + player.current_bet
                    
                    print(f"[DEBUG] Raise calculation: current_bet={table.game.current_bet}, min_raise_increment={min_raise_increment}, min_raise_to={min_raise_to}, max_possible={max_possible}")
                    
                    if max_possible >= min_raise_to:
                        # Can make a legal raise
                        raise_amount = min_raise_to
                        poker_action = "raise"
                    elif max_possible > table.game.current_bet:
                        # Can't make min raise but can go all-in
                        raise_amount = max_possible
                        poker_action = "raise"  # All-in raise
                    else:
                        # Can't raise, must call or fold
                        poker_action = "call" if to_call > 0 and player.stack >= to_call else "fold"
                        raise_amount = 0
            else:
                poker_action = "fold"
                raise_amount = 0
        
        # Execute action with error handling
        prev_stack = player.stack
        try:
            table.game.step(poker_action, raise_amount)
            
            # Validate state consistency after action execution
            if hasattr(table.game, '_validate_state_consistency'):
                table.game._validate_state_consistency(f"after tournament action {poker_action} by {player.name}")
                
        except Exception as e:
            # If game step fails, return penalty and continue
            print(f"ERROR: Game step failed for {player.name}: {type(e).__name__}: {e}")
            print(f"[DEBUG] Failed action: {poker_action}, raise_amount: {raise_amount}")
            print(f"[DEBUG] Player stack: {player.stack}, current_bet: {player.current_bet}")
            print(f"[DEBUG] Game current_bet: {table.game.current_bet}, last_raise: {table.game.last_raise_amount}")
            print(f"[DEBUG] Big blind: {table.game.big_blind}, to_call calculated as: {max(0, table.game.current_bet - player.current_bet)}")
            obs = self._get_obs()
            return obs, -10, False, False, {"action_mask": self.legal_action_mask()}
        
        # Calculate comprehensive tournament reward
        reward = self._calculate_reward(player, prev_stack)
        
        # Check for eliminations after each action (players can bust mid-hand)
        self._clean_elimination_order()  # Clean up inconsistent elimination tracking
        self._update_elimination_order()
        
        # Check if hand is over
        if table.game.hand_over:
            table.hands_played += 1
            self.hands_played_this_level += 1
            self.total_hands_played += 1
            
            # Start new hand if table still active
            if table.get_active_player_count() >= 2:
                try:
                    blind_level = self.blinds_schedule[self.current_blind_level]
                    sb, bb, ante = blind_level  # All levels are 3-tuples after normalization
                    table.game.small_blind = sb
                    table.game.big_blind = bb
                    table.game.ante = ante
                    table.game.reset_for_new_hand(is_first_hand=False)
                except Exception as e:
                    print(f"Error resetting hand: {e}")
                    table.is_active = False
            else:
                table.is_active = False
            
            # Check for eliminations
            self._clean_elimination_order()  # Clean up inconsistent elimination tracking
            self._update_elimination_order()

            # Table balancing: after hand at this table
            try:
                self.balance_table(table.table_id)
            except Exception as e:
                print(f"[DEBUG] Error in table balancing: {e}")

            # Check blind increases
            self._increase_blinds_if_needed()

            # Move to next active table
            next_table = self._select_next_active_table()
            if next_table is not None:
                self.active_table_id = next_table
        
        # Check if tournament is finished
        terminated = self._tournament_finished()
        
        obs = self._get_obs()
        info = {"action_mask": self.legal_action_mask()}
        
        return obs, reward, terminated, False, info
    
    def render(self, mode="human"):
        """Render current tournament state"""
        print(f"\n=== Multi-Table Tournament Status ===")
        print(f"Blind Level: {self.current_blind_level + 1} ({self.blinds_schedule[self.current_blind_level]})")
        print(f"Hands this level: {self.hands_played_this_level}/{self.hands_per_blind_level}")
        print(f"Active Tables: {len(self._get_active_tables())}")
        print(f"Players remaining: {len([p for p in self.all_players if p.stack > 0])}")
        print(f"Players eliminated: {len(self.elimination_order)}")
        
        for table in self._get_active_tables():
            active_count = table.get_active_player_count()
            print(f"  Table {table.table_id}: {active_count} players")
    
    def get_tournament_stats(self) -> Dict:
        """Get comprehensive tournament statistics"""
        remaining_players = [p for p in self.all_players if p.stack > 0]
        active_tables = self._get_active_tables()
        
        return {
            "total_players": self.total_players,
            "remaining_players": len(remaining_players),
            "eliminated_players": len(self.elimination_order),
            "active_tables": len(active_tables),
            "current_blind_level": self.current_blind_level + 1,
            "blinds": self.blinds_schedule[self.current_blind_level],
            "hands_played": self.total_hands_played,
            "average_stack": np.mean([p.stack for p in remaining_players]) if remaining_players else 0,
            "chip_leader": max(remaining_players, key=lambda p: p.stack).name if remaining_players else None,
            "chip_leader_stack": max(remaining_players, key=lambda p: p.stack).stack if remaining_players else 0
        }
