# poker-ai/engine/game.py

from engine.cards import Deck
from engine.player import Player
from engine.hand_evaluator import hand_rank
from utils.enums import GameMode  # <-- Added import

class PokerGame:
    def __init__(self, players, starting_stack=1000, small_blind=10, big_blind=20, game_mode=GameMode.AI_VS_AI):  # <-- Added game_mode param with default
        if len(players) < 2:
            raise ValueError("Need at least two players to start the game.")
        self.players = players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.game_mode = game_mode  # <-- Store game_mode
        self.deck = None
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0  # Current highest bet to call
        self.active_players = []  # Players still in hand (not folded)
        self.dealer_position = 0  # Index of dealer; rotates each hand

    def reset_for_new_hand(self):
        self.deck = Deck()
        self.deck.shuffle()
        
        self.pot = 0
        self.community_cards = []
        self.current_bet = 0
        
        for player in self.players:
            player.reset_for_new_hand()
            player.stack = max(player.stack, 0)
        
        self.active_players = [p for p in self.players if p.stack > 0]
        self.players_who_posted_blinds = set()
        if len(self.active_players) < 2:
            raise RuntimeError("Not enough players with chips to continue.")

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

    def betting_round(self):
        print("\n--- Betting round ---")

        active_players = [p for p in self.players if p.in_hand and not p.all_in]
        players_acted = set()
        last_raiser = None
        self.current_bet = max(p.current_bet for p in self.players)

        while True:
            all_acted = all(p in players_acted for p in active_players)
            unresolved_bets = any(
                p.current_bet != self.current_bet and not p.all_in for p in active_players
            )

            if all_acted and not unresolved_bets:
                break

            for player in self.players_in_order(start=self.dealer_position + 1):
                if not player.in_hand or player.all_in or player in players_acted:
                    continue

                to_call = self.current_bet - player.current_bet

                print(f"Acting: {player.name}, in_hand={player.in_hand}, current_bet={player.current_bet}")

                # Auto-check only for bots who posted blind and already matched current bet
                if (
                    player.name in self.players_who_posted_blinds
                    and player.current_bet == self.current_bet
                    and not player.is_human
                ):
                    print(f"{player.name} (blind) checks")
                    players_acted.add(player)
                    continue

                if player.is_human:
                    self.prompt_human_action(player, to_call)
                else:
                    self.take_ai_action(player, to_call)

                players_acted.add(player)
                self.current_bet = max(p.current_bet for p in self.players)

            print(f"Current pot: {self.pot}")
            for p in self.players:
                print(
                    f"{p.name}: stack={p.stack} current_bet={p.current_bet} in_hand={p.in_hand}"
                )

    def prompt_human_action(self, player, to_call):
        while True:
            print(f"\nYour turn, {player.name}. Your cards: {player.hole_cards}")
            print(f"Community cards: {self.community_cards}")
            print(f"Current pot: {self.pot}, to call: {to_call}")
            print(f"Stack: {player.stack}")

            options = ["fold"]
            if to_call == 0:
                options.append("check")
            else:
                options.append("call")
            if player.stack > to_call:
                options.append("raise <amount>")

            action = input(f"Choose action {options}: ").strip().lower()
            parts = action.split()

            if action == "fold":
                player.in_hand = False
                return

            elif action == "check" and to_call == 0:
                print(f"{player.name} checks")
                return

            elif action == "call" and to_call > 0:
                if to_call > player.stack:
                    to_call = player.stack  # All-in call
                player.bet_chips(to_call)
                self.pot += to_call
                print(f"{player.name} calls {to_call}")
                return

            elif parts[0] == "raise" and len(parts) == 2 and parts[1].isdigit():
                raise_amount = int(parts[1])
                total_bet = to_call + raise_amount
                if total_bet > player.stack:
                    print("You don't have enough chips to raise that amount.")
                    continue
                player.bet_chips(total_bet)
                self.pot += total_bet
                print(f"{player.name} raises by {raise_amount} to {player.current_bet}")
                return

            print("Invalid action. Try again.")


    def all_players_equal_bet(self):
        bets = [p.current_bet for p in self.active_players if p.in_hand]
        return len(set(bets)) == 1

    def print_stacks_and_pot(self):
        print(f"Current pot: {self.pot}")
        for p in self.players:
            print(f"{p.name}: stack={p.stack} current_bet={p.current_bet} in_hand={p.in_hand}")

    def play_hand(self):
        self.reset_for_new_hand()
        self.rotate_dealer()
        print(f"\n{'='*30}")
        print(f"Dealer is {self.players[self.dealer_position].name}")
        print(f"{'='*30}")

        self.post_blinds()
        self.deal_hole_cards()

        print("\n--- Pre-flop ---")
        first_to_act = (self.dealer_position + 3) % len(self.players)
        self.betting_round(first_to_act)
        self.players_who_posted_blinds = set()
        self.print_stacks_and_pot()

        if len([p for p in self.active_players if p.in_hand]) == 1:
            self.end_hand()
            return

        print("\n--- Flop ---")
        self.deal_community_cards(3)
        self.reset_bets()
        self.betting_round((self.dealer_position + 1) % len(self.players))
        self.print_stacks_and_pot()

        if len([p for p in self.active_players if p.in_hand]) == 1:
            self.end_hand()
            return

        print("\n--- Turn ---")
        self.deal_community_cards(1)
        self.reset_bets()
        self.betting_round((self.dealer_position + 1) % len(self.players))
        self.print_stacks_and_pot()

        if len([p for p in self.active_players if p.in_hand]) == 1:
            self.end_hand()
            return

        print("\n--- River ---")
        self.deal_community_cards(1)
        self.reset_bets()
        self.betting_round((self.dealer_position + 1) % len(self.players))
        self.print_stacks_and_pot()

        self.showdown()
        print("\nFinal stacks:")
        for p in self.players:
            print(f"{p.name}: {p.stack} chips")
        print(f"{'='*30}\n")


    def reset_bets(self):
        for player in self.players:
            player.current_bet = 0
        self.current_bet = 0

    def end_hand(self):
        remaining = [p for p in self.active_players if p.in_hand]
        if remaining:
            winner = remaining[0]
            ...
        print(f"\nHand ends early, {winner.name} wins the pot of {self.pot} chips!")
        winner.stack += self.pot

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

if __name__ == "__main__":
    from engine.player import Player
    from utils.enums import GameMode  # <-- added import here

    alice = Player("Alice")
    bob = Player("Bob")

    # Pass game_mode explicitly if you want AI vs AI
    game = PokerGame([alice, bob], game_mode=GameMode.AI_VS_AI)
    game.play_hand()
