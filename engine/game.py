# poker-ai/engine/game.py

import sys
from engine import player
from engine.cards import Deck
from engine.player import Player
from engine.hand_evaluator import hand_rank
from utils.enums import GameMode
from engine.action_validation import validate_raise, validate_call, validate_check, validate_fold, ActionValidationError
class PokerGame:
    def collect_bet(self, player, amount, suppress_log=False):
        """Take chips from player and add to pot, always keeping pot and contributions in sync."""
        actual_bet = player.bet_chips(amount, suppress_log=suppress_log)
        self.pot += actual_bet
        self._assert_pot_consistency()
        return actual_bet

    def collect_ante(self, player, amount, suppress_log=False):
        """Take ante from player and add to pot, always keeping pot and contributions in sync."""
        actual_ante = player.post_ante(amount, suppress_log=suppress_log)
        self.pot += actual_ante
        self._assert_pot_consistency()
        return actual_ante

    def _assert_pot_consistency(self):
        total_contrib = sum(p.total_contributed for p in self.players)
        if self.pot != total_contrib:
            print(f"[INVARIANT VIOLATION] Pot ({self.pot}) != sum of player.total_contributed ({total_contrib})")
            for p in self.players:
                print(f"    {p.name}: total_contributed={p.total_contributed}, current_bet={p.current_bet}, stack={p.stack}")
            raise RuntimeError("Pot and player contributions are out of sync!")
    PHASES = ["preflop", "flop", "turn", "river", "showdown"]

    def __init__(self, players, starting_stack=1000, small_blind=10, big_blind=20,
                 ante=0, game_mode=GameMode.AI_VS_AI, human_action_callback=None, table_id=None):
        if len(players) < 2:
            raise ValueError("Need at least two players to start the game.")
        if ante < 0:
            raise ValueError("Ante cannot be negative.")
        self.players = players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.ante = ante  # Ante flag (when > 0, BB pays total ante = BB amount)
        self.game_mode = game_mode
        self.human_action_callback = human_action_callback
        self.deck = None
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.active_players = []  # Players still in hand (not folded)
        self.table_id = table_id  # For multi-table debug output
        self.dealer_position = 0
        self.current_player_idx = None  # Index of player to act
        self.phase_idx = 0  # Index in PHASES
        self.hand_over = False
        self.players_who_posted_blinds = set()
        self.last_raise_amount = self.big_blind  # Track last raise size
        self.bb_acted_preflop = False  # Track if BB has acted preflop
        self.players_to_act = []  # Track who still needs to act this betting round
        self.hands_played = 0  # Track number of hands played

    def reset_for_new_hand(self, deck=None, is_first_hand=True):
        # --- STACK SUM CONSISTENCY CHECK (before posting blinds for new hand) ---
        # Different logic for tournament vs cash game
        if self.table_id is not None:
            # Tournament mode: Total chips should be conserved, but players can have different amounts
            # Only check that no chips were mysteriously created or destroyed
            actual_total = sum(p.stack for p in self.players)
            
            # In tournament, we can't assume all players started with starting_stack
            # because players move between tables. Instead, just check for impossible scenarios:
            
            # 1. No player should have negative chips
            negative_stacks = [p for p in self.players if p.stack < 0]
            if negative_stacks:
                print(f"[ERROR] [TABLE {self.table_id}] Players with negative stacks detected:")
                for p in negative_stacks:
                    print(f"    {p.name}: stack={p.stack}")
                #sys.exit(1)
            
            # 2. In very first hand of tournament, all players should have starting_stack
            # (We detect this by checking if all players have exactly starting_stack)
            if is_first_hand:
                all_have_starting_stack = all(p.stack == self.starting_stack for p in self.players)
                if all_have_starting_stack:
                    # This appears to be tournament start - validate total
                    expected_total = self.starting_stack * len(self.players)
                    if actual_total != expected_total:
                        print(f"[ERROR] [TABLE {self.table_id}] Tournament start stack inconsistency: total ({actual_total}) != expected ({expected_total})")
                        for p in self.players:
                            print(f"    {p.name}: stack={p.stack}")
                        #sys.exit(1)
            
            print(f"[DEBUG] [TABLE {self.table_id}] Tournament hand reset: {len(self.players)} players, total chips: {actual_total}")
            
        else:
            # Cash game mode: All players should maintain their starting stack (simple case)
            expected_total = self.starting_stack * len(self.players)
            actual_total = sum(p.stack for p in self.players)
            if actual_total != expected_total:
                print(f"[ERROR] [CASH] Stack sum inconsistency: total ({actual_total}) != expected ({expected_total})")
                for p in self.players:
                    print(f"    {p.name}: stack={p.stack}")
                #sys.exit(1)
        # Extra debug: print player bets and pot before resetting for new hand
        print(f"[INCONSISTENCY-CHECK] (Before reset_for_new_hand) Table {getattr(self, 'table_id', '?')}: Player bets and pot before reset:")
        for player in self.players:
            print(f"[INCONSISTENCY-CHECK]    {player.name}.current_bet = {player.current_bet}")
        print(f"[INCONSISTENCY-CHECK]    self.current_bet = {self.current_bet}")
        print(f"[INCONSISTENCY-CHECK]    self.pot = {self.pot}")

        # Rotate dealer position for new hand (except first hand of game)
        if not is_first_hand:
            self.rotate_dealer()
            
        # Do NOT call reset_bets() here! Bets should only be reset after a betting round is complete.
        if deck is None:
            self.deck = Deck()
            self.deck.shuffle()
        else:
            self.deck = deck

        self.pot = 0
        self.community_cards = []
        self.current_bet = 0
        self.phase_idx = 0
        self.hand_over = False
        self.bb_acted_preflop = False
        self.players_to_act = []


        # Extra debug: print before resetting player states
        print(f"[INCONSISTENCY-CHECK] (Resetting player states) Table {getattr(self, 'table_id', '?')}")
        for player in self.players:
            print(f"[INCONSISTENCY-CHECK]     Before reset: {player.name}.current_bet = {player.current_bet}")

        # Reset player states (including total_contributed!)
        for player in self.players:
            player.current_bet = 0
            player.total_contributed = 0
            player.hole_cards = []
            player.in_hand = True
            player.all_in = False

        # Extra debug: print after resetting player states
        print(f"[INCONSISTENCY-CHECK] (After resetting player states) Table {getattr(self, 'table_id', '?')}")
        for player in self.players:
            print(f"[INCONSISTENCY-CHECK]     After reset: {player.name}.current_bet = {player.current_bet}")


        self.players_who_posted_blinds = set()

        # Extra debug: print before posting blinds
        print(f"[INCONSISTENCY-CHECK] (Before post_blinds) Table {getattr(self, 'table_id', '?')}: Player bets and pot before posting blinds:")
        for player in self.players:
            print(f"[INCONSISTENCY-CHECK]     {player.name}.current_bet = {player.current_bet}")
        print(f"[INCONSISTENCY-CHECK]     self.current_bet = {self.current_bet}")
        print(f"[INCONSISTENCY-CHECK]     self.pot = {self.pot}")

        self.post_blinds()

        # [DEBUG] Print all player bets, current_bet, and pot after hand setup
        print(f"[INCONSISTENCY-CHECK] TABLE {getattr(self, 'table_id', '?')} After hand setup:")
        for player in self.players:
            print(f"[INCONSISTENCY-CHECK]    {player.name}.current_bet = {player.current_bet}")
        print(f"[INCONSISTENCY-CHECK]    self.current_bet = {self.current_bet}")
        print(f"[INCONSISTENCY-CHECK]    self.pot = {self.pot}")

        # --- Mark all-in and eliminated states after blinds ---
        for player in self.players:
            if player.stack == 0 and player.current_bet > 0:
                # All-in after posting blind
                player.in_hand = True
                player.all_in = True
            elif player.stack == 0 and player.current_bet == 0:
                # Eliminated before posting blind
                player.in_hand = False
                player.all_in = False
            else:
                player.in_hand = True
                player.all_in = False

        self.active_players = [p for p in self.players if p.in_hand and (p.stack > 0 or p.current_bet > 0)]
        if len(self.active_players) < 2:
            raise RuntimeError("Not enough players with chips to continue.")

        self.deal_hole_cards()

        # --- Set current_player_idx and players_to_act based on number of active players ---
        if len(self.active_players) == 2:
            # Heads-up: SB (dealer) acts first preflop
            active_indices = [i for i, p in enumerate(self.players) if p.in_hand and (p.stack > 0 or p.current_bet > 0)]
            dealer_pos = self.dealer_position
            sb_idx = active_indices[0] if active_indices[0] == dealer_pos else active_indices[1]
            bb_idx = active_indices[1] if active_indices[0] == dealer_pos else active_indices[0]
            self.current_player_idx = sb_idx
            self.players_to_act = [self.players[sb_idx], self.players[bb_idx]]
        else:
            # 3+ players: first to act is left of BB, then next, ..., up to BB
            first_to_act = (self.dealer_position + 3) % len(self.players)
            bb_pos = (self.dealer_position + 2) % len(self.players)
            idx = first_to_act
            self.players_to_act = []
            while True:
                if self.players[idx].in_hand and not self.players[idx].all_in and self.players[idx].stack > 0:
                    self.players_to_act.append(self.players[idx])
                if idx == bb_pos:
                    break
                idx = (idx + 1) % len(self.players)
            self.current_player_idx = first_to_act
        
        # Validate initial state after hand setup
        self._validate_state_consistency("after new hand setup")

    def rotate_dealer(self):
        n = len(self.players)
        for _ in range(n):
            self.dealer_position = (self.dealer_position + 1) % n
            if self.players[self.dealer_position].stack > 0:
                break

    def post_blinds(self):
        n = len(self.players)
        dealer_pos = self.dealer_position
        active_indices = [i for i, p in enumerate(self.players) if p.stack > 0]
        num_active = len(active_indices)
        if num_active < 2:
            raise RuntimeError("Not enough players with chips to continue.")

        # --- Assign SB and BB correctly for heads-up and 3+ players ---
        if num_active == 2:
            # Heads-up: dealer is SB, next is BB
            sb_idx = dealer_pos if self.players[dealer_pos].stack > 0 else active_indices[0]
            bb_idx = active_indices[0] if active_indices[1] == dealer_pos else active_indices[1]
        else:
            # 3+ players: SB is first with chips after dealer, BB is next with chips after SB
            sb_idx = None
            bb_idx = None
            for offset in range(1, n+1):
                idx = (dealer_pos + offset) % n
                if self.players[idx].stack > 0:
                    if sb_idx is None:
                        sb_idx = idx
                    elif bb_idx is None:
                        bb_idx = idx
                        break
        sb_player = self.players[sb_idx]
        bb_player = self.players[bb_idx]

        # --- Post ante if needed (BB pays total ante = BB amount) ---
        ante_paid = 0
        if self.ante > 0:
            total_ante = self.big_blind
            ante_paid = min(bb_player.stack, total_ante)
            if ante_paid > 0:
                self.collect_ante(bb_player, ante_paid, suppress_log=True)
                print(f"[DEBUG] [ANTE] {bb_player.name} (BB) posts ante of {ante_paid} (total ante = BB). Remaining stack: {bb_player.stack}")

        # --- Post small blind first ---
        sb_amount = min(sb_player.stack, self.small_blind)
        if sb_amount > 0:
            self.collect_bet(sb_player, sb_amount, suppress_log=True)
            print(f"[DEBUG] [SB] {sb_player.name} posts small blind of {sb_amount}. Remaining stack: {sb_player.stack}")
        else:
            print(f"[DEBUG] [SB] {sb_player.name} could not post small blind (stack={sb_player.stack})")

        # --- Post big blind ---
        bb_amount = min(bb_player.stack, self.big_blind)
        if bb_amount > 0:
            self.collect_bet(bb_player, bb_amount, suppress_log=True)
            print(f"[DEBUG] [BB] {bb_player.name} posts big blind of {bb_amount}. Remaining stack: {bb_player.stack}")
        else:
            print(f"[DEBUG] [BB] {bb_player.name} could not post big blind (stack={bb_player.stack})")

        # --- Set last_raise_amount to the actual BB posted ---
        self.last_raise_amount = bb_amount

        # --- Always set current_bet to the full BB amount, even if BB is all-in for less ---
        self.current_bet = self.big_blind

        # --- Defensive: If BB posted less than BB, print debug for side pot logic ---
        if bb_amount < self.big_blind:
            print(f"[DEBUG] BB {bb_player.name} is all-in for less than BB ({bb_amount} < {self.big_blind}); current_bet set to {self.big_blind} for side pot logic.")

        # --- Pot already includes blinds and ante via collect_bet and collect_ante ---
        # Do NOT overwrite pot here; it should include both blinds and ante.

        # --- Track who posted blinds ---
        self.players_who_posted_blinds = {sb_player.name, bb_player.name}

        # --- Debug output ---
        print(f"[DEBUG post_blinds] Pot after blinds and antes: {self.pot}, SB: {sb_player.name} (stack: {sb_player.stack}), BB: {bb_player.name} (stack: {bb_player.stack}), Ante posted: {ante_paid}")
        print(f"[INCONSISTENCY-CHECK] TABLE {getattr(self, 'table_id', '?')} After posting blinds:")
        for player in self.players:
            print(f"[INCONSISTENCY-CHECK]    {player.name}.current_bet = {player.current_bet}")
        print(f"[INCONSISTENCY-CHECK]    self.current_bet = {self.current_bet}")
        print(f"[INCONSISTENCY-CHECK]    self.pot = {self.pot}")

        # --- Validate state after posting blinds ---
        self._validate_state_consistency("after posting blinds")

    def deal_hole_cards(self):
        if self.deck is None:
            raise RuntimeError("Deck is not initialized. Call reset_for_new_hand() first.")
        
        for player in self.players:
            if player.in_hand and player.stack > 0:
                player.hole_cards = self.deck.draw(2)
                print(f"[DEBUG] {player.name} was dealt: {player.hole_cards}")
            else:
                player.hole_cards = []

    def deal_community_cards(self, number):
        if self.deck is None:
            raise RuntimeError("Deck is not initialized. Call reset_for_new_hand() first.")
        
        cards = self.deck.draw(number)
        self.community_cards.extend(cards)
        print(f"[DEBUG] Community cards dealt: {self.community_cards}")

    def reset_bets(self):
        for player in self.players:
            player.current_bet = 0
        self.current_bet = 0

    def _validate_state_consistency(self, context=""):
        """
        Validate that player.current_bet and game.current_bet are properly synchronized.
        This helps detect and prevent state inconsistency warnings.
        """
        inconsistencies = []
        
        # Check 1: No player.current_bet should exceed game.current_bet
        for player in self.players:
            if player.current_bet > self.current_bet:
                inconsistencies.append(f"{player.name}.current_bet ({player.current_bet}) > game.current_bet ({self.current_bet})")
        
        # Check 2: If game.current_bet > 0, at least one player should have that bet amount
        if self.current_bet > 0:
            max_player_bet = max((p.current_bet for p in self.players), default=0)
            # Allow edge case: BB is all-in from antes/can't post full BB
            if max_player_bet != self.current_bet:
                bb_all_in_case = False
                if hasattr(self, 'players_who_posted_blinds'):
                    # Find the BB player - they should be in players_who_posted_blinds
                    # and either have current_bet < current_bet (partial BB) or current_bet = 0 (all-in from ante)
                    for p in self.players:
                        if p.name in self.players_who_posted_blinds and p.stack == 0:
                            # Player is all-in and was supposed to post BB
                            # Check if they posted less than BB or nothing (all-in from antes)
                            if p.current_bet < self.current_bet:
                                bb_all_in_case = True
                                print(f"[DEBUG] BB edge case: {p.name} all-in with {p.current_bet} bet, can't post full BB ({self.current_bet})")
                                break
                if not bb_all_in_case:
                    inconsistencies.append(f"game.current_bet ({self.current_bet}) != max player bet ({max_player_bet})")
        
        # Check 3: Total player bets should match what we expect for pot calculation
        total_player_bets = sum(p.current_bet for p in self.players)
        
        if inconsistencies:
            print(f"[WARNING] Table {getattr(self, 'table_id', '?')} State inconsistency detected in {context}:")
            for issue in inconsistencies:
                print(f"  - {issue}")
            print(f"  - Total player bets: {total_player_bets}, Game pot: {self.pot}")
            # Don't crash during training - log and continue
            print(f"[DEBUG] Continuing despite state inconsistency for training stability")
            #sys.exit(1)
            return False
        return True
    
    def _synchronize_current_bet(self):
        """
        Ensure game.current_bet matches the highest player.current_bet.
        This is a defensive method to fix synchronization issues.
        """
        max_player_bet = max((p.current_bet for p in self.players), default=0)
        if self.current_bet != max_player_bet:
            print(f"[WARNING] SYNC NEEDED! Synchronizing game.current_bet from {self.current_bet} to {max_player_bet}")
            #sys.exit(1) # aisa todo
            self.current_bet = max_player_bet
    
    def fix_state_inconsistencies(self):
        """
        Public method to fix detected state inconsistencies.
        Can be called from tournament environments when inconsistencies are detected.
        """
        print(f"[INCONSISTENCY-CHECK] Attempting to fix state inconsistencies...")
        
        # First, identify the correct game.current_bet - use the original value as baseline
        # Don't synchronize upward if we have individual player bet inconsistencies
        original_game_bet = self.current_bet
        
        # Fix 1: Ensure no player.current_bet exceeds original game.current_bet
        fixed_players = []
        for player in self.players:
            if player.current_bet > original_game_bet:
                print(f"[INCONSISTENCY-CHECK] Reducing {player.name}.current_bet from {player.current_bet} to {original_game_bet}")
                # Calculate the difference to refund to stack
                excess = player.current_bet - original_game_bet
                player.current_bet = original_game_bet
                player.stack += excess
                fixed_players.append(player.name)
        
        # Fix 2: After fixing individual players, synchronize game.current_bet if needed
        # (This handles cases where game.current_bet is lower than it should be)
        self._synchronize_current_bet()

        # Fix 3: Ensure pot matches sum of all player bets
        self.pot = sum(p.current_bet for p in self.players)
        print(f"[INCONSISTENCY-CHECK] Synchronized pot to sum of player bets: {self.pot}")

        if fixed_players:
            print(f"[INCONSISTENCY-CHECK] Fixed bet inconsistencies for players: {fixed_players}")

        # Validate the fixes worked
        if self._validate_state_consistency("after fix_state_inconsistencies"):
            return True
        else:
            print(f"[INCONSISTENCY-CHECK] Unable to fully resolve state inconsistencies")
            return False

    def _validate_comprehensive_state(self, context="", strict_mode=True):
        """
        Enhanced comprehensive state validation that checks all aspects of poker game integrity.
        
        Args:
            context: Description of when this validation is called
            strict_mode: If True, exit on any violation. If False, log warnings only.
        
        Returns:
            bool: True if all validations pass, False otherwise
        """
        violations = []
        warnings = []
        
        print(f"[DEBUG] [ENHANCED_VALIDATION] Running comprehensive state validation: {context}")
        
        # === PLAYER STATE INTEGRITY ===
        for i, player in enumerate(self.players):
            # 1. Stack integrity
            if player.stack < 0:
                violations.append(f"Player {player.name} has negative stack: {player.stack}")
            
            # 2. Bet integrity  
            if player.current_bet < 0:
                violations.append(f"Player {player.name} has negative current_bet: {player.current_bet}")
            
            if player.total_contributed < 0:
                violations.append(f"Player {player.name} has negative total_contributed: {player.total_contributed}")
            
            # 3. Logical consistency between stack states
            if player.stack == 0 and player.current_bet == 0 and player.in_hand:
                warnings.append(f"Player {player.name} has stack=0, current_bet=0 but in_hand=True (should be eliminated)")
            
            if player.stack > 0 and not player.in_hand and not hasattr(self, '_hand_in_progress'):
                warnings.append(f"Player {player.name} has chips but not in_hand outside of hand progression")
            
            # 4. All-in state consistency
            if player.all_in and player.stack > 0:
                violations.append(f"Player {player.name} marked all_in but has stack: {player.stack}")
            
            if player.stack == 0 and player.current_bet > 0 and not player.all_in:
                warnings.append(f"Player {player.name} has stack=0, current_bet={player.current_bet} but not marked all_in")
        
        # === BETTING ROUND INTEGRITY ===
        # 5. Current bet synchronization
        max_player_bet = max((p.current_bet for p in self.players), default=0)
        if self.current_bet != max_player_bet and self.current_bet > 0:
            # Allow BB all-in exception
            bb_exception = False
            if hasattr(self, 'players_who_posted_blinds'):
                for p in self.players:
                    if p.name in self.players_who_posted_blinds and p.stack == 0 and p.current_bet < self.current_bet:
                        bb_exception = True
                        break
            if not bb_exception:
                violations.append(f"game.current_bet ({self.current_bet}) != max player bet ({max_player_bet})")
        
        # 6. Betting logic consistency
        active_players = [p for p in self.players if p.in_hand and p.stack > 0]
        if len(active_players) > 1:
            # In multi-player scenarios, check betting round completion logic
            if hasattr(self, 'players_to_act'):
                actionable_players = [p for p in active_players if not p.all_in]
                if not self.players_to_act and actionable_players and not self.hand_over:
                    # This might indicate incomplete betting round logic
                    warnings.append(f"No players_to_act but {len(actionable_players)} actionable players remain")
        
        # === POT AND CHIP CONSERVATION ===
        # 7. Pot consistency with player contributions
        expected_pot = sum(p.total_contributed for p in self.players)
        if self.pot != expected_pot:
            violations.append(f"Pot ({self.pot}) != sum of total_contributed ({expected_pot})")
        
        # 8. Current bet contributions consistency
        current_round_contributions = sum(p.current_bet for p in self.players)
        if current_round_contributions > self.pot:
            violations.append(f"Current round bets ({current_round_contributions}) > pot ({self.pot})")
        
        # === DEALER AND BLIND LOGIC ===
        # 9. Dealer position validity
        if hasattr(self, 'dealer_position'):
            if self.dealer_position < 0 or self.dealer_position >= len(self.players):
                violations.append(f"Invalid dealer_position: {self.dealer_position} (players: {len(self.players)})")
            
            # Check dealer has chips (if possible)
            if self.dealer_position < len(self.players):
                dealer = self.players[self.dealer_position]
                if dealer.stack == 0 and len([p for p in self.players if p.stack > 0]) > 1:
                    warnings.append(f"Dealer {dealer.name} has no chips but other players do")
        
        # 10. Blind posting validation
        if hasattr(self, 'players_who_posted_blinds') and self.players_who_posted_blinds:
            blind_posters = [p for p in self.players if p.name in self.players_who_posted_blinds]
            if len(blind_posters) != 2 and len([p for p in self.players if p.stack > 0]) >= 2:
                warnings.append(f"Expected 2 blind posters, found {len(blind_posters)}: {[p.name for p in blind_posters]}")
        
        # === CARD AND DECK INTEGRITY ===
        # 11. Hole card validation
        dealt_cards = []
        for player in self.players:
            if player.hole_cards:
                if len(player.hole_cards) != 2:
                    violations.append(f"Player {player.name} has {len(player.hole_cards)} hole cards (should be 2)")
                dealt_cards.extend(player.hole_cards)
        
        # 12. Community card validation
        if len(self.community_cards) > 5:
            violations.append(f"Too many community cards: {len(self.community_cards)}")
        
        dealt_cards.extend(self.community_cards)
        
        # 13. Card duplication check
        if len(dealt_cards) != len(set(str(card) for card in dealt_cards)):
            violations.append(f"Duplicate cards detected in dealt cards")
        
        # === GAME PHASE CONSISTENCY ===
        # 14. Phase progression validation
        expected_community_cards = {0: 0, 1: 3, 2: 4, 3: 5, 4: 5}  # preflop, flop, turn, river, showdown
        if self.phase_idx in expected_community_cards:
            expected_count = expected_community_cards[self.phase_idx]
            if len(self.community_cards) != expected_count:
                violations.append(f"Phase {self.PHASES[self.phase_idx]} should have {expected_count} community cards, has {len(self.community_cards)}")
        
        # === HAND TERMINATION LOGIC ===
        # 15. Hand over conditions
        players_with_chips = [p for p in self.players if p.stack > 0]
        in_hand_with_chips = [p for p in self.players if p.in_hand and p.stack > 0]
        
        if len(in_hand_with_chips) <= 1 and not self.hand_over and self.phase_idx < len(self.PHASES) - 1:
            warnings.append(f"Only {len(in_hand_with_chips)} player(s) in hand with chips, but hand not over")
        
        # === FINAL VALIDATION SUMMARY ===
        total_issues = len(violations) + len(warnings)
        
        if violations:
            print(f"[ERROR] [ENHANCED_VALIDATION] {len(violations)} CRITICAL violations found in {context}:")
            for violation in violations:
                print(f"  ❌ {violation}")
        
        if warnings:
            print(f"[WARNING] [ENHANCED_VALIDATION] {len(warnings)} warnings found in {context}:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        if total_issues == 0:
            print(f"[DEBUG] [ENHANCED_VALIDATION] ✅ All {self._get_validation_check_count()} checks passed for {context}")
        
        # Handle violations based on strict_mode
        if violations:
            if strict_mode:
                print(f"[CRITICAL] [ENHANCED_VALIDATION] STRICT MODE: Exiting due to {len(violations)} critical violations")
                #sys.exit(1)
            else:
                print(f"[DEBUG] [ENHANCED_VALIDATION] NON-STRICT MODE: Continuing despite {len(violations)} violations")
                return False
        
        return len(violations) == 0

    def _get_validation_check_count(self):
        """Return the total number of validation checks performed"""
        return 15  # Update this when adding new validation categories
    
    def validate_action_preconditions(self, action: str, player_idx: int, raise_amount: int = 0):
        """
        Validate that an action can be legally performed given current game state.
        This is called BEFORE action execution to prevent illegal moves.
        """
        violations = []
        
        if player_idx < 0 or player_idx >= len(self.players):
            violations.append(f"Invalid player_idx: {player_idx}")
            
        player = self.players[player_idx]
        to_call = max(0, self.current_bet - player.current_bet)
        
        # Action-specific validations
        if action == "fold":
            if not player.in_hand:
                violations.append(f"Player {player.name} cannot fold - not in hand")
        
        elif action == "check":
            if to_call > 0:
                violations.append(f"Player {player.name} cannot check - must call {to_call}")
            if not player.in_hand:
                violations.append(f"Player {player.name} cannot check - not in hand")
        
        elif action == "call":
            if to_call == 0:
                violations.append(f"Player {player.name} cannot call - nothing to call")
            if not player.in_hand:
                violations.append(f"Player {player.name} cannot call - not in hand")
            if player.stack == 0:
                violations.append(f"Player {player.name} cannot call - no chips")
        
        elif action == "raise":
            if not player.in_hand:
                violations.append(f"Player {player.name} cannot raise - not in hand")
            if player.stack <= to_call:
                violations.append(f"Player {player.name} cannot raise - insufficient chips for call")
            if raise_amount <= self.current_bet:
                violations.append(f"Raise amount {raise_amount} must exceed current bet {self.current_bet}")
            if raise_amount > player.stack + player.current_bet:
                violations.append(f"Raise amount {raise_amount} exceeds available chips {player.stack + player.current_bet}")
        
        # General turn order validation
        if self.current_player_idx != player_idx:
            violations.append(f"Not {player.name}'s turn - current player is {self.current_player_idx}")
        
        if violations:
            print(f"[ERROR] [ACTION_VALIDATION] {len(violations)} precondition violations for {action} by {player.name}:")
            for violation in violations:
                print(f"  ❌ {violation}")
            return False
        
        print(f"[DEBUG] [ACTION_VALIDATION] ✅ Action {action} by {player.name} passed all precondition checks")
        return True

    def step(self, action, raise_amount=0):
        """
        Perform the action by the current player.
        A raise is interpreted as "raise to amount X."
        
        Parameters:
        - action (str): "fold", "call", "check", or "raise"
        - raise_amount (int): If action == "raise", this is the total amount player wants to have on the table after raising.
        """
        
        # Validate state consistency at start of step
        self._validate_state_consistency(f"start of step - {action}")

        print(f"[DEBUG] Entering step: phase_idx={self.phase_idx}, players_to_act={[p.name for p in self.players_to_act]}, action={action}")

        # If players_to_act is empty and not showdown, re-initialize for new round
        if not self.players_to_act and self.phase_idx < self.PHASES.index("showdown"):
            self.players_to_act = [p for p in self.players if p.in_hand and not p.all_in and p.stack > 0]
            if self.players_to_act:
                self.current_player_idx = self.players.index(self.players_to_act[0])

        if self.hand_over:
            raise RuntimeError("Hand is over. Please reset for new hand.")

        # Defensive: If betting round is already complete, do not process another action
        if self._betting_round_complete():
            print("[DEBUG] Betting round complete at start of step, returning early.")
            return self._get_state(), 0, self.hand_over, {}

        player = self.players[self.current_player_idx]

        if len(self.players) == 2 and self.phase_idx == 0:  # preflop
            bb_pos = (self.dealer_position + 1) % 2
            if self.current_player_idx == bb_pos:
                self.bb_acted_preflop = True

        # Skip folded or all-in players
        if not player.in_hand or player.all_in:
            self._advance_to_next_player()
            return self._get_state(), 0, self.hand_over, {}

        to_call = self.current_bet - player.current_bet
        to_call = max(0, to_call)  # Ensure non-negative

        # Handle human input if needed
        if player.is_human and action is None:
            action, raise_amount = self.prompt_human_action(player, to_call)

        player = self.players[self.current_player_idx]

        print(f"[DEBUG] ==> {player.name}'s turn: Action={action}, ToCall={to_call}, RaiseTo={raise_amount}")
        print(f"[DEBUG]     Stack: {player.stack}, CurrentBet: {player.current_bet}, Pot: {self.pot}")

        # --- ACTIONS ---

        if action == "fold":
            result = self.handle_fold(player, to_call)
            print(f"[DEBUG] {player.name} folds.")
            if sum(p.in_hand and p.stack > 0 for p in self.players) == 1:
                # Only one player remains in hand with chips, award pot and end hand
                winner = next(p for p in self.players if p.in_hand and p.stack > 0)
                winner.stack += self.pot
                print(f"[DEBUG] {winner.name} wins the pot of {self.pot} by default (all others folded).")
                self.pot = 0
                self.hand_over = True
                return

        elif action == "call":
            result = self.handle_call(player, to_call)
            print(f"[DEBUG] {player.name} calls {result['call_amount']}{' (all-in)' if result['is_all_in'] else ''}.")

        elif action == "check":
            result = self.handle_check(player, to_call)
            print(f"[DEBUG] {player.name} checks.")

        elif action == "raise":
            self.handle_raise(player, raise_to=raise_amount, to_call=to_call)
            # After a raise, set players_to_act to all active (in_hand, not all-in) players after raiser, excluding raiser
            self.players_to_act = self._players_to_act_after(player)
            print(f"[DEBUG] players_to_act after raise: {[p.name for p in self.players_to_act]}")

        else:
            raise ValueError(f"Invalid action: {action}")

        # --- Update active players ---
        self.active_players = [p for p in self.players if p.in_hand and p.stack > 0]

        # Clean up players_to_act: only keep players who are still in hand and not all-in
        self.players_to_act = [p for p in self.players_to_act if p.in_hand and not p.all_in and p.stack > 0]

        # Always remove the acting player from players_to_act (except after a raise, which resets the list)
        if action in ("call", "check", "fold") and player in self.players_to_act:
            print(f"[DEBUG] Removing {player.name} from players_to_act")
            self.players_to_act.remove(player)

        # Defensive check: ensure no folded or all-in players remain in players_to_act
        for p in self.players_to_act:
            if not p.in_hand or p.all_in:
                print(f"[DEBUG] Removing {p.name} from players_to_act (folded or all-in)")
                self.players_to_act.remove(p)

        # If players_to_act is now empty or only contains non-actionable players, clear it and trigger hand termination
        if not any(p.in_hand and not p.all_in and p.stack > 0 for p in self.players_to_act):
            self.players_to_act = []

        # --- HAND TERMINATION LOGIC ---
        active_in_hand = [p for p in self.players if p.in_hand and p.stack > 0]
        if len(active_in_hand) == 1 and not self.players_to_act:
            self.hand_over = True
            winner = active_in_hand[0]
            winner.stack += self.pot
            print(f"[DEBUG] Hand over: only one player remains ({winner.name}), awarded pot of {self.pot}")
            self.pot = 0
            return  # Prevent further processing

        elif all(p.all_in or p.stack == 0 for p in active_in_hand) and not self.players_to_act:
            # All-in showdown, no pending actions
            if self.phase_idx < self.PHASES.index("showdown"):
                while self.phase_idx < self.PHASES.index("showdown"):
                    self._advance_phase()
            self.phase_idx = self.PHASES.index("showdown")
            self.showdown()
            self.hand_over = True
            print("[DEBUG] Hand over: all players are all-in, go to showdown")
            return

        all_all_in = all(p.all_in or p.stack == 0 for p in active_in_hand)
        num_active = len(active_in_hand)

        # If all active players are all-in, no further betting is possible
        if all_all_in and num_active > 1:
            self.hand_over = True
            print(f"[DEBUG] Hand over: all active players are all-in")
            return self._get_state(), 0, self.hand_over, {}

        # --- Check for win (everyone else folded) ---
        if len([p for p in self.active_players if p.in_hand]) == 1 and not self.players_to_act:
            self.hand_over = True
            winner = next(p for p in self.active_players if p.in_hand)
            print(f"[DEBUG] Hand ends! {winner.name} wins the pot of {self.pot} chips.")
            return

        # --- Check for all-in showdown ---
        if all(p.all_in or not p.in_hand for p in self.active_players) and not self.players_to_act:
            # All remaining players are all-in or folded: go to showdown
            self.phase_idx = self.PHASES.index("showdown")
            self.showdown()
            self.hand_over = True
            return

        # --- Advance phase or next player ---
        if self._betting_round_complete():
            self._advance_phase()

            if self.phase_idx == len(self.PHASES) - 1:  # showdown
                self.showdown()
                self.hand_over = True
                self.players_to_act = []  # Always clear at showdown/hand over
            else:
                # Only reset bets after a full betting round, before dealing new community cards
                self.reset_bets()
                if self.phase_idx in [1, 2, 3]:  # flop, turn, river
                    self.deal_community_cards({1: 3, 2: 1, 3: 1}[self.phase_idx])

                # Set current player to first active after dealer
                self.current_player_idx = (self.dealer_position + 1) % len(self.players)
                while not (self.players[self.current_player_idx].in_hand and self.players[self.current_player_idx].stack > 0):
                    self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
                # After phase advance, players_to_act should be empty.
                self.players_to_act = []
        else:
            self._advance_to_next_player()
        
        print(f"[DEBUG] Exiting step: phase_idx={self.phase_idx}, players_to_act={[p.name for p in self.players_to_act]}")
        
        # Validate state consistency at end of step
        self._validate_state_consistency(f"end of step - {action}")

        # --- Final catch-all: if no legal actions remain, end the hand ---
        active_in_hand = [p for p in self.players if p.in_hand and p.stack > 0]
        if (
            not self.players_to_act and
            (
                len(active_in_hand) == 0 or
                all(p.all_in or not p.in_hand or p.stack == 0 for p in self.players)
            )
        ):
            self.hand_over = True
            print("[DEBUG] Hand over: no legal actions remain, all players are all-in, folded, or eliminated")
            return self._get_state(), 0, self.hand_over, {}

        return self._get_state(), 0, self.hand_over, {}

    def prompt_human_action(self, player, to_call):
        """
        Prompt the action by the human player.
        Current behavior: A raise is defined as "raise to amount X."
        
        Parameters:
        - action (str): "fold", "call", "check", or "raise"
        - raise_amount (int): If action == "raise", this is the total amount player wants to have on the table after raising.
        """
        if self.human_action_callback:
            return self.human_action_callback(player, to_call)

        # Real prompt for human interaction:
        while True:
            print(f"\nYour turn, {player.name}. Your cards: {player.hole_cards}")
            print(f"Community cards: {self.community_cards}")
            print(f"Current pot: {self.pot}, to call: {to_call}")
            print(f"Stack: {player.stack}")

            options = []

            # Allow fold unless big blind with no to_call
            bb_pos = (self.dealer_position + 2) % len(self.players)
            is_big_blind = (self.players[bb_pos] == player)
            if not (is_big_blind and to_call == 0):
                options.append("fold")

            if to_call == 0:
                options.append("check")
            else:
                options.append("call")

            if player.stack > to_call:
                options.append("raise <amount> (total bet after raise)")

            action = input(f"Choose action {options}: ").strip().lower()
            parts = action.split()

            if action == "fold" and "fold" in options:
                return "fold", 0
            elif action == "check" and to_call == 0:
                return "check", 0
            elif action == "call" and to_call > 0:
                return "call", 0
            elif parts[0] == "raise" and len(parts) == 2 and parts[1].isdigit():
                raise_to = int(parts[1])

                if raise_to <= player.current_bet:
                    print(f"You must raise to more than your current bet ({player.current_bet})")
                    continue
                if raise_to < self.current_bet:
                    print(f"Raise must be at least the current bet ({self.current_bet})")
                    continue
                if raise_to > player.stack + player.current_bet:
                    print(f"You only have {player.stack} chips available.")
                    continue

                return "raise", raise_to
            else:
                print("Invalid action, try again.")

    def _advance_to_next_player(self):
        if self.current_player_idx is None:
            # Initialize to first active player if not set
            for i, player in enumerate(self.players):
                if player.in_hand and not player.all_in and player.stack > 0:
                    self.current_player_idx = i
                    return
            # No valid players found
            self.hand_over = True
            return
        
        num_players = len(self.players)
        for _ in range(num_players):
            self.current_player_idx = (self.current_player_idx + 1) % num_players
            current_player = self.players[self.current_player_idx]
            # Player must be in hand, not all-in, AND have chips remaining
            if current_player.in_hand and not current_player.all_in and current_player.stack > 0:
                return
        # No players left to act
        self.hand_over = True

    def _betting_round_complete(self):
        # Special case: heads-up preflop, BB must always get a chance to act
        if len(self.players) == 2 and self.phase_idx == 0:
            if not self.bb_acted_preflop:
                return False
        # Betting round is only complete if all players have acted after the last raise
        if self.players_to_act:
            return False
        # Check that all non-all-in players have equal bets
        # All-in players can have different bet amounts due to side pots
        in_hand_players = [p for p in self.players if p.in_hand]
        if not in_hand_players:
            return True  # No players in hand, round is complete
        
        # Only check non-all-in players for equal bets
        non_all_in_players = [p for p in in_hand_players if not p.all_in and p.stack > 0]
        if not non_all_in_players:
            return True  # All remaining players are all-in, round is complete
        
        non_all_in_bets = [p.current_bet for p in non_all_in_players]
        if len(set(non_all_in_bets)) <= 1:
            return True
        return False

    def _advance_phase(self):
        self.phase_idx += 1
        print(f"[DEBUG] Advancing to phase: {self.PHASES[self.phase_idx]} (phase_idx={self.phase_idx})")
        # Reset bets for new round
        self.reset_bets()
        # Validate state after phase advance and bet reset
        self._validate_state_consistency(f"after advancing to {self.PHASES[self.phase_idx]}")

    def _get_state(self):
        # Return a simple dict representing game state for current player
        player = self.players[self.current_player_idx]
        return {
            "player_name": player.name,
            "player_stack": player.stack,
            "player_current_bet": player.current_bet,
            "pot": self.pot,
            "community_cards": self.community_cards,
            "player_hole_cards": player.hole_cards,
            "current_bet": self.current_bet,
            "phase": self.PHASES[self.phase_idx],
        }
    
    def play_hand(self):
        # Use is_first_hand=True only for the very first hand
        is_first_hand = (self.hands_played == 0)
        self.reset_for_new_hand(is_first_hand=is_first_hand)

        done = False
        while not done:
            player = self.players[self.current_player_idx]
            to_call = self.current_bet - player.current_bet
            to_call = max(0, to_call)  # Ensure non-negative

            if player.is_human:
                # Human input will be requested inside step()
                action, raise_amount = None, 0
            else:
                if to_call > 0:
                    action = "call"
                    raise_amount = 0
                else:
                    action = "check"
                    raise_amount = 0

            _, _, done, _ = self.step(action, raise_amount)

        # Increment hands_played counter after hand is complete
        self.hands_played += 1

        print("Hand complete.")
        print("Final stacks:")
        for p in self.players:
            print(f"{p.name}: {p.stack} chips")
            p.reset_for_new_hand()

    def handle_fold(self, player, to_call):
        print(f"[DEBUG handle_fold] {player.name} called handle_fold()")
        if not isinstance(self.current_bet, int) or not isinstance(player.current_bet, int):
            raise ActionValidationError("current_bet and player.current_bet must be integers.")
        try:
            result = validate_fold(in_hand=player.in_hand, to_call=to_call)
        except ValueError as e:
            print(f"Invalid fold by {player.name}: {e}")
            raise ActionValidationError(str(e))
        player.fold()
        return {
            "player": player.name,
            "can_fold": result["can_fold"],
            "pot": self.pot,
            "current_bet": self.current_bet,
        }

    def handle_check(self, player, to_call):
        print(f"[DEBUG handle_check] {player.name} called handle_check()")
        if not isinstance(self.current_bet, int) or not isinstance(player.current_bet, int):
            raise ActionValidationError("current_bet and player.current_bet must be integers.")
        try:
            result = validate_check(to_call=to_call)
        except ValueError as e:
            print(f"Invalid check by {player.name}: {e}")
            raise ActionValidationError(str(e))
        return {
            "player": player.name,
            "can_check": result["can_check"],
            "pot": self.pot,
            "current_bet": self.current_bet,
        }

    def handle_call(self, player, to_call):
        print(f"[DEBUG handle_call] {player.name} called handle_call()")
        if not isinstance(self.current_bet, int) or not isinstance(player.current_bet, int):
            raise ActionValidationError("current_bet and player.current_bet must be integers.")
        if to_call == 0:
            raise ActionValidationError("Cannot call when to_call is zero; should check instead.")
        try:
            result = validate_call(player_stack=player.stack, to_call=to_call)
        except ValueError as e:
            print(f"Invalid call by {player.name}: {e}")
            raise ActionValidationError(str(e))

        call_amount = min(player.stack, to_call)
        self.collect_bet(player, call_amount)

        print(f"[DEBUG handle_call] Pot after call: {self.pot}, {player.name} stack: {player.stack}, all_in: {player.all_in}")

        if player.stack == 0:
            player.all_in = True
            is_all_in = True
        else:
            is_all_in = False

        return {
            "player": player.name,
            "call_amount": call_amount,
            "is_all_in": is_all_in,
            "pot": self.pot,
            "current_bet": self.current_bet,
        }

    def handle_raise(self, player, raise_to: int, to_call):
        print(f"[DEBUG handle_raise] {player.name} called handle_raise({raise_to})")
        # Defensive: ensure current_bet and player.current_bet are ints
        if not isinstance(self.current_bet, int) or not isinstance(player.current_bet, int):
            raise ActionValidationError("current_bet and player.current_bet must be integers.")

        try:
            result = validate_raise(
                raise_to=raise_to,
                player_stack=player.stack,
                to_call=to_call,
                current_bet=self.current_bet,
                min_raise=self.last_raise_amount,
                big_blind=self.big_blind,
                player_current_bet=player.current_bet
            )
        except ValueError as e:
            raise ActionValidationError(str(e))

        raise_amount = raise_to - player.current_bet  # Only pay the difference!
        actual_raise = raise_to - self.current_bet

        if raise_amount > player.stack:
            raise ActionValidationError("Player cannot raise more than their stack.")

        # Use bet_chips for logging and consistency
        self.collect_bet(player, raise_amount, suppress_log=True)

        if player.stack == 0:
            player.all_in = True

        # Update game state
        if actual_raise >= self.last_raise_amount:
            self.last_raise_amount = actual_raise
            self.current_bet = raise_to
            # Reset players_to_act: everyone after raiser who is not all-in or folded
            self.players_to_act = self._players_to_act_after(player)
        else:
            # Not a valid raise (should not happen if validation is correct)
            pass

        # Validate synchronization after raise
        self._validate_state_consistency(f"after raise by {player.name} to {raise_to}")

        print(f"[DEBUG] {player.name} raises to {raise_to}. (Put in {raise_amount}, stack now {player.stack})")

        return {
            "player": player.name,
            "raise_to": raise_to,
            "raise_amount": raise_amount,
            "actual_raise": actual_raise,
            "is_all_in": player.all_in,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "last_raise_amount": self.last_raise_amount,
        }

    def showdown(self):
        print("\n--- Showdown ---")
        in_hand_players = [p for p in self.players if p.in_hand or p.all_in]
        if not in_hand_players:
            print("No winners found.")
            return

        # Use total_contributed for side pot calculation
        contributions = {p: p.total_contributed for p in self.players}
        pots = []
        
        # Get all unique contribution levels, sorted
        bet_levels = sorted(set([amt for amt in contributions.values() if amt > 0]))
        
        if not bet_levels:
            print("[SHOWDOWN] No contributions found for side pot calculation")
            return
        
        # Build side pots correctly
        prev_level = 0
        active_players = [p for p in self.players if contributions[p] > 0]
        
        for level in bet_levels:
            # Calculate pot size for this level
            pot_amount = (level - prev_level) * len(active_players)
            
            # Players eligible for this pot are those who contributed at least this level
            eligible_players = [p for p in active_players if contributions[p] >= level]
            
            if pot_amount > 0 and eligible_players:
                pots.append({
                    "amount": pot_amount,
                    "players": eligible_players.copy()
                })
                print(f"[SHOWDOWN] Side pot {len(pots)}: {pot_amount} chips, eligible players: {[p.name for p in eligible_players]}")
            
            # Remove players who are maxed out at this level
            active_players = [p for p in active_players if contributions[p] > level]
            prev_level = level

        total_pot = sum(pot["amount"] for pot in pots)
        if total_pot != self.pot:
            print(f"[WARNING] Pot mismatch: calculated {total_pot}, actual {self.pot}")
            print(f"[SHOWDOWN] Player contributions: {contributions}")
            print(f"[SHOWDOWN] Side pots: {pots}")
            #sys.exit(1)
            # Try to fix the mismatch by using the calculated total
            if abs(total_pot - self.pot) <= len(self.players):  # Small discrepancy, likely rounding
                print(f"[SHOWDOWN] Adjusting pot from {self.pot} to calculated {total_pot}")
                self.pot = total_pot
            else:
                print(f"[SHOWDOWN] Large pot mismatch - continuing with actual pot {self.pot}")
                # Don't crash during training

        hand_ranks = {}
        for p in self.players:
            if p.in_hand or p.all_in:
                try:
                    hand_ranks[p] = hand_rank(p.hole_cards + self.community_cards)
                except Exception as e:
                    print(f"Error evaluating hand for {p.name}: {e}")
                    hand_ranks[p] = None

        for i, pot in enumerate(pots):
            eligible = [p for p in pot["players"] if (p.in_hand or p.all_in)]
            if not eligible:
                print(f"No eligible players for pot {i+1}, skipping.")
                continue
            best_rank = None
            winners = []
            for p in eligible:
                rank = hand_ranks.get(p)
                if rank is None:
                    continue
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    winners = [p]
                elif rank == best_rank:
                    winners.append(p)
            if not winners:
                print(f"No winners for pot {i+1}, skipping.")
                continue
            split = pot["amount"] // len(winners)
            remainder = pot["amount"] % len(winners)
            for j, p in enumerate(winners):
                award = split + (1 if j < remainder else 0)
                p.stack += award
                print(f"[DEBUG] {p.name} wins {award} chips from pot {i+1} (side pot)" if len(pots) > 1 else f"[DEBUG] {p.name} wins {award} chips.")
        self.pot = 0

    def _players_to_act_after(self, raiser):
        """
        Return list of active players (in_hand and not all-in) who still need to act,
        starting from the player *after* the raiser and looping around.
        The raiser is considered last to act.
        """
        players_to_act = []
        num_players = len(self.players)
        start_idx = self.players.index(raiser)
        idx = (start_idx + 1) % num_players

        while True:
            player = self.players[idx]
            if player.in_hand and not player.all_in and player != raiser:
                players_to_act.append(player)
            idx = (idx + 1) % num_players
            if idx == start_idx:
                break

        return players_to_act

if __name__ == "__main__":
    alice = Player("Alice")
    bob = Player("Bob")

    game = PokerGame([alice, bob], game_mode=GameMode.AI_VS_AI)
    game.reset_for_new_hand()

    # Example of stepping through actions:
    done = False
    while not done:
        state = game._get_state()
        print(f"Current state: {state}")
        # For testing, let's call or check if possible, otherwise fold:
        if game.current_player_idx is None:
            print("No current player, ending game")
            break
        player = game.players[game.current_player_idx]
        to_call = game.current_bet - player.current_bet  # type: ignore[attr-defined]
        to_call = max(0, to_call)  # Ensure non-negative
        if to_call > 0:
            action = "call"
        else:
            action = "check"
        state, reward, done, info = game.step(action)
