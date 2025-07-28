# poker-ai/engine/game.py

from engine import player
from engine.cards import Deck
from engine.player import Player
from engine.hand_evaluator import hand_rank
from utils.enums import GameMode
from engine.action_validation import validate_raise, validate_call, validate_check, validate_fold, ActionValidationError

class PokerGame:
    PHASES = ["preflop", "flop", "turn", "river", "showdown"]

    def __init__(self, players, starting_stack=1000, small_blind=10, big_blind=20,
                 ante=0, game_mode=GameMode.AI_VS_AI, human_action_callback=None):
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

        # Reset player states
        for player in self.players:
            player.current_bet = 0
            player.hole_cards = []
            player.in_hand = True
            player.all_in = False

        self.players_who_posted_blinds = set()
        self.post_blinds()

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

    def rotate_dealer(self):
        n = len(self.players)
        for _ in range(n):
            self.dealer_position = (self.dealer_position + 1) % n
            if self.players[self.dealer_position].stack > 0:
                break

    def post_blinds(self):
        active_indices = [i for i, p in enumerate(self.players) if p.stack > 0]
        num_active = len(active_indices)
        if num_active < 2:
            raise RuntimeError("Not enough players with chips to continue.")

        dealer_pos = self.dealer_position

        if num_active == 2:
            # Heads-up: dealer is SB, next is BB
            dealer_pos = self.dealer_position
            sb_idx = active_indices[0] if active_indices[0] == dealer_pos else active_indices[1]
            bb_idx = active_indices[1] if active_indices[0] == dealer_pos else active_indices[0]
        else:
            # More than 2: SB is next active after dealer, BB is next after SB
            dealer_pos = self.dealer_position
            sb_idx = None
            bb_idx = None
            for offset in range(1, len(self.players) + 1):
                idx = (dealer_pos + offset) % len(self.players)
                if self.players[idx].stack > 0:
                    if sb_idx is None:
                        sb_idx = idx
                    elif bb_idx is None:
                        bb_idx = idx
                        break

        sb_player = self.players[sb_idx]
        bb_player = self.players[bb_idx]

        # Post antes first (BB pays total ante equal to big blind amount)
        total_ante = 0
        if self.ante > 0:
            # In modern tournaments, total ante often equals the big blind
            # BB pays the total ante for the entire table
            total_ante = self.big_blind  # Total ante = BB amount
            
            ante_paid = min(bb_player.stack, total_ante)
            if ante_paid > 0:
                bb_player.bet_chips(ante_paid, suppress_log=True)
                self.pot += ante_paid
                print(f"[DEBUG] {bb_player.name} posts ante of {ante_paid} (total ante = BB). Remaining stack: {bb_player.stack}")

        # Post small blind
        sb_amount = min(sb_player.stack, self.small_blind)
        sb_player.bet_chips(sb_amount, suppress_log=True)
        print(f"[DEBUG] {sb_player.name} posts small blind of {sb_amount}. Remaining stack: {sb_player.stack}")
        self.pot += sb_amount

        # Post big blind
        bb_amount = min(bb_player.stack, self.big_blind)
        bb_player.bet_chips(bb_amount, suppress_log=True)
        print(f"[DEBUG] {bb_player.name} posts big blind of {bb_amount}. Remaining stack: {bb_player.stack}")
        self.pot += bb_amount

        print(f"[DEBUG post_blinds] Pot after blinds and antes: {self.pot}, SB stack: {sb_player.stack}, BB stack: {bb_player.stack}")

        self.current_bet = bb_amount
        self.last_raise_amount = bb_amount

        # Note: current_bet was already updated by bet_chips() calls above
        # No need to add again as bet_chips() already handled it

        self.players_who_posted_blinds = {sb_player.name, bb_player.name}

    def deal_hole_cards(self):
        for player in self.players:
            if player.in_hand and player.stack > 0:
                player.hole_cards = self.deck.draw(2)
                print(f"[DEBUG] {player.name} was dealt: {player.hole_cards}")
            else:
                player.hole_cards = []

    def deal_community_cards(self, number):
        cards = self.deck.draw(number)
        self.community_cards.extend(cards)
        print(f"[DEBUG] Community cards dealt: {self.community_cards}")

    def reset_bets(self):
        for player in self.players:
            player.current_bet = 0
        self.current_bet = 0

    def step(self, action, raise_amount=0):
        """
        Perform the action by the current player.
        A raise is interpreted as "raise to amount X."
        
        Parameters:
        - action (str): "fold", "call", "check", or "raise"
        - raise_amount (int): If action == "raise", this is the total amount player wants to have on the table after raising.
        """

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

    def _players_to_act_after(self, raiser):
        """
        Returns a list of all in-hand, not-all-in players who must act after a raise,
        in table order, starting with the next player after the raiser, wrapping around,
        and excluding the raiser.
        """
        players = []
        n = len(self.players)
        raiser_idx = self.players.index(raiser)
        idx = (raiser_idx + 1) % n
        while idx != raiser_idx:
            p = self.players[idx]
            if p.in_hand and not p.all_in:
                players.append(p)
            idx = (idx + 1) % n
        return players

    def _advance_to_next_player(self):
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
        player.bet_chips(call_amount)
        self.pot += call_amount

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
        player.bet_chips(raise_amount, suppress_log=True)
        self.pot += raise_amount

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
        bet_levels = sorted(set([amt for amt in contributions.values() if amt > 0]))
        prev = 0
        remaining_players = set(self.players)
        for level in bet_levels:
            pot_players = [p for p in remaining_players if contributions[p] >= level]
            if not pot_players:
                continue
            pot_size = (level - prev) * len(pot_players)
            pots.append({"amount": pot_size, "players": pot_players.copy()})
            prev = level
            for p in list(remaining_players):
                if contributions[p] == level:
                    remaining_players.remove(p)

        total_pot = sum(pot["amount"] for pot in pots)
        if total_pot != self.pot:
            print(f"[WARNING] Pot mismatch: calculated {total_pot}, actual {self.pot}")

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
    from engine.player import Player
    from utils.enums import GameMode

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
        player = game.players[game.current_player_idx]
        to_call = game.current_bet - player.current_bet
        to_call = max(0, to_call)  # Ensure non-negative
        if to_call > 0:
            action = "call"
        else:
            action = "check"
        state, reward, done, info = game.step(action)
