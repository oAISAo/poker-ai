# poker-ai/engine/game.py

from engine.cards import Deck
from engine.player import Player
from engine.hand_evaluator import hand_rank
from utils.enums import GameMode
from engine.action_validation import validate_raise, validate_call, ActionValidationError

class PokerGame:
    PHASES = ["preflop", "flop", "turn", "river", "showdown"]

    def __init__(self, players, starting_stack=1000, small_blind=10, big_blind=20,
                game_mode=GameMode.AI_VS_AI, human_action_callback=None):
        if len(players) < 2:
            raise ValueError("Need at least two players to start the game.")
        self.players = players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.game_mode = game_mode
        self.human_action_callback = human_action_callback  # <-- add this line
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

    def reset_for_new_hand(self, deck=None):
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

        for player in self.players:
            player.reset_for_new_hand()
            player.stack = max(player.stack, 0)

        self.active_players = [p for p in self.players if p.stack > 0]
        if len(self.active_players) < 2:
            raise RuntimeError("Not enough players with chips to continue.")

        self.players_who_posted_blinds = set()
        self.post_blinds()
        self.deal_hole_cards()

        self.current_player_idx = (self.dealer_position + 3) % len(self.players)
        self.last_raise_amount = self.big_blind

    def rotate_dealer(self):
        self.dealer_position = (self.dealer_position + 1) % len(self.players)

    def post_blinds(self):
        sb_pos = (self.dealer_position + 1) % len(self.players)
        bb_pos = (self.dealer_position + 2) % len(self.players)

        sb_player = self.players[sb_pos]
        bb_player = self.players[bb_pos]

        sb_amount = min(sb_player.stack, self.small_blind)
        sb_player.bet_chips(sb_amount, suppress_log=True)
        print(f"{sb_player.name} posts small blind of {sb_amount}. Remaining stack: {sb_player.stack}")
        self.pot += sb_amount

        bb_amount = min(bb_player.stack, self.big_blind)
        bb_player.bet_chips(bb_amount, suppress_log=True)
        print(f"{bb_player.name} posts big blind of {bb_amount}. Remaining stack: {bb_player.stack}")
        self.pot += bb_amount

        self.current_bet = bb_amount
        self.last_raise_amount = bb_amount

        sb_player.current_bet = sb_amount
        bb_player.current_bet = bb_amount

        self.players_who_posted_blinds = {sb_player.name, bb_player.name}

    def deal_hole_cards(self):
        for player in self.active_players:
            hole_cards = self.deck.draw(2)
            player.deal_hole_cards(hole_cards)
            print(f"{player.name} was dealt: {[str(card) for card in hole_cards]}")

    def deal_community_cards(self, number):
        cards = self.deck.draw(number)
        self.community_cards.extend(cards)
        print(f"Community cards dealt: {self.community_cards}")

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
        if self.hand_over:
            raise RuntimeError("Hand is over. Please reset for new hand.")

        player = self.players[self.current_player_idx]

        # Skip folded or all-in players
        if not player.in_hand or player.all_in:
            self._advance_to_next_player()
            return self._get_state(), 0, self.hand_over, {}

        to_call = self.current_bet - player.current_bet

        # Handle human input if needed
        if player.is_human and action is None:
            action, raise_amount = self.prompt_human_action(player, to_call)

        print(f"\n==> {player.name}'s turn: Action={action}, ToCall={to_call}, RaiseTo={raise_amount}")
        print(f"    Stack: {player.stack}, CurrentBet: {player.current_bet}, Pot: {self.pot}")

        # --- ACTIONS ---

        if action == "fold":
            player.fold()
            print(f"{player.name} folds.")

        elif action == "call":
            result = self.handle_call(player)
            print(f"{player.name} calls {result['call_amount']}{' (all-in)' if result['is_all_in'] else ''}.")

        elif action == "check":
            if to_call > 0:
                raise ValueError("Cannot check when to_call > 0")
            print(f"{player.name} checks.")

        elif action == "raise":
            self.handle_raise(player, raise_to=raise_amount)

        else:
            raise ValueError(f"Invalid action: {action}")

        # --- Update active players ---
        self.active_players = [p for p in self.players if p.in_hand and p.stack > 0]

        # --- Check for win (everyone else folded) ---
        if len(self.active_players) == 1:
            self.hand_over = True
            winner = self.active_players[0]
            winner.stack += self.pot
            print(f"\n🏆 Hand ends! {winner.name} wins the pot of {self.pot} chips.")
            return self._get_state(), 1, self.hand_over, {}

        # --- Advance phase or next player ---
        if self._betting_round_complete():
            self._advance_phase()

            if self.phase_idx == len(self.PHASES) - 1:
                self.showdown()
                self.hand_over = True
            else:
                self.reset_bets()
                if self.phase_idx in [1, 2, 3]:  # flop, turn, river
                    self.deal_community_cards({1: 3, 2: 1, 3: 1}[self.phase_idx])

            # Set current player to first active after dealer
            self.current_player_idx = (self.dealer_position + 1) % len(self.players)
            while not self.players[self.current_player_idx].in_hand:
                self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        else:
            self._advance_to_next_player()

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
                options.append("raise <amount>")

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
        num_players = len(self.players)
        for _ in range(num_players):
            self.current_player_idx = (self.current_player_idx + 1) % num_players
            if self.players[self.current_player_idx].in_hand and not self.players[self.current_player_idx].all_in:
                return
        # No players left to act
        self.hand_over = True

    def _betting_round_complete(self):
        # Betting round complete if all active players have equal bets or are all-in
        active_bets = [p.current_bet for p in self.active_players if p.in_hand]
        if len(set(active_bets)) <= 1:
            return True
        return False

    def _advance_phase(self):
        self.phase_idx += 1
        print(f"Advancing to phase: {self.PHASES[self.phase_idx]}")

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
        self.reset_for_new_hand()
        self.rotate_dealer()

        done = False
        while not done:
            player = self.players[self.current_player_idx]
            to_call = self.current_bet - player.current_bet

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

        print("Hand complete.")
        print("Final stacks:")
        for p in self.players:
            print(f"{p.name}: {p.stack} chips")
            p.reset_for_new_hand()

    def handle_call(self, player):
        to_call = self.current_bet - player.current_bet
        if to_call == 0:
            raise ActionValidationError("Cannot call when to_call is zero; should check instead.")
        try:
            result = validate_call(player_stack=player.stack, to_call=to_call)
        except ValueError as e:
            print(f"Invalid call by {player.name}: {e}")
            raise

        call_amount = result["call_amount"]
        is_all_in = result["is_all_in"]

        player.stack -= call_amount
        player.current_bet += call_amount
        self.pot += call_amount

        if player.stack == 0:
            player.all_in = True

        return {
            "player": player.name,
            "call_amount": call_amount,
            "is_all_in": is_all_in,
            "pot": self.pot,
            "current_bet": self.current_bet,
        }

    def handle_raise(self, player, raise_to: int):
        to_call = self.current_bet - player.current_bet
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
            print(f"Invalid raise by {player.name}: {e}")
            raise

        raise_amount = raise_to - player.current_bet
        actual_raise = raise_to - self.current_bet

        player.stack -= raise_amount
        player.current_bet = raise_to
        self.pot += raise_amount

        # Update game state
        if actual_raise >= self.last_raise_amount:
            self.last_raise_amount = actual_raise
            self.last_aggressor = player
            self.current_bet = raise_to
            self.active_players = self._players_to_act_after(player)
        else:
            # All-in smaller raise: current_bet and last_raise stay unchanged
            if player in self.active_players:
                self.active_players.remove(player)

        # Optionally return result for testing/logging
        return {
            "player": player.name,
            "raise_to": raise_to,
            "raise_amount": raise_amount,
            "actual_raise": actual_raise,
            "is_all_in": result.is_all_in,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "last_raise_amount": self.last_raise_amount,
        }

    def showdown(self):
        print("\n--- Showdown ---")
        winners = [p for p in self.active_players if p.in_hand]
        if not winners:
            print("No winners found.")
            return

        best_rank = None
        best_players = []

        for player in winners:
            full_hand = player.hole_cards + self.community_cards
            rank, best_hand = hand_rank(full_hand)
            best_hand_str = ' '.join(str(card) for card in best_hand)
            print(f"{player.name}: {rank} with [{best_hand_str}]")

            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_players = [player]
            elif rank == best_rank:
                best_players.append(player)

        split_pot = self.pot // len(best_players)
        for winner in best_players:
            print(f"{winner.name} wins {split_pot} chips")
            winner.stack += split_pot

        print("Hand complete.")

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

        # Optionally append raiser at the end if they need to act last
        # (depends on your game logic; sometimes raiser acts last after everyone else)
        # players_to_act.append(raiser)

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
        if to_call > 0:
            action = "call"
        else:
            action = "check"
        state, reward, done, info = game.step(action)
