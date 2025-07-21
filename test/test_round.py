import random
from engine.game import PokerGame
from engine.player import Player
from utils.enums import GameMode
import pytest
from engine.cards import Card  # Assuming you have a Card class for representing cards

def test_bug_raise_resets_betting_round(monkeypatch):
    class TestPlayer(Player):
        def __init__(self, name):
            super().__init__(name, stack=1000, is_human=False)
            self.action_sequence = ["call", "raise 50", "call", "check"]
            self.action_index = 0

        def decide_action(self, to_call, community_cards):
            if self.action_index < len(self.action_sequence):
                action = self.action_sequence[self.action_index]
                self.action_index += 1
                return "raise" if action.startswith("raise") else action
            return "check"

    players = [TestPlayer("P1"), TestPlayer("P2"), TestPlayer("P3")]
    game = PokerGame(players, game_mode=GameMode.AI_VS_AI)

    game.reset_for_new_hand()
    game.dealer_position = 0
    game.post_blinds()
    game.deal_hole_cards()

    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    # Start from player after big blind (dealer + 3)
    game.current_player_idx = 1

    done = False
    while not done:
        player = game.players[game.current_player_idx]
        to_call = game.current_bet - player.current_bet

        action = player.decide_action(to_call, game.community_cards)
        raise_amount = 50 if action == "raise" else 0

        _, _, done, _ = game.step(action, raise_amount=raise_amount)

    total_bets = sum(p.current_bet for p in players)
    assert total_bets >= game.current_bet * len(players)
    assert all(p.in_hand for p in players)


def test_human_raise_with_mocked_cards():
    from engine.game import PokerGame, GameMode
    from engine.player import Player

    # Mock deck to fix hole cards and community cards
    class MockDeck:
        def __init__(self):
            # Human hole cards (strong hand)
            self.cards = [
                Card('A', '♠'), Card('K', '♠'),  # Human hole cards
                Card('2', '♦'), Card('7', '♣'),  # Bot hole cards
                # Community cards (very strong for human)
                Card('A', '♦'), Card('K', '♦'), Card('Q', '♦'),
                Card('J', '♦'), Card('T', '♦')
            ]
            self.index = 0

        def shuffle(self):
            pass  # No shuffle for deterministic order

        def draw(self, n):
            drawn = self.cards[self.index:self.index + n]
            self.index += n
            return drawn

    # Human action: raise once, then call/check
    def mock_human_action_factory():
        raised_once = [False]
        def action(player, to_call):
            if not raised_once[0]:
                raised_once[0] = True
                return ("raise", 50)
            else:
                if to_call > 0:
                    return ("call", 0)
                else:
                    return ("check", 0)
        return action

    mock_human_action = mock_human_action_factory()

    players = [
        Player("You", stack=1000, is_human=True),
        Player("Bot", stack=1000, is_human=False)
    ]
    game = PokerGame(players, game_mode=GameMode.HUMAN_VS_AI, human_action_callback=mock_human_action)

    game.reset_for_new_hand(deck=MockDeck())
    game.dealer_position = 0  # (optional, if you want to fix dealer)
    game.play_hand()

    # After raising once 50 chips, human stack should have decreased by at least 50 chips from blinds + raise
    # Since human has the best hand, stack should be > 1000 after winning the pot
    assert players[0].stack > 1000, f"Expected human stack > 1000 but got {players[0].stack}"

    # Pot should be at least the raise amount + blinds
    assert game.pot >= 50

    # Bot should lose chips
    assert players[1].stack < 1000, f"Expected bot stack < 1000 but got {players[1].stack}"

def test_bug_all_in_sets_flag(monkeypatch):
    class AllInTestPlayer(Player):
        def decide_action(self, to_call, community_cards):
            if to_call > self.stack:
                return "fold"
            elif to_call == self.stack:
                return "call"  # All in
            else:
                return "call"

    ai1 = AllInTestPlayer("AI1", stack=20, is_human=False)
    ai2 = AllInTestPlayer("AI2", stack=1000, is_human=False)
    game = PokerGame([ai1, ai2], game_mode=GameMode.AI_VS_AI)

    game.reset_for_new_hand()
    game.dealer_position = 0
    game.post_blinds()
    game.deal_hole_cards()

    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    # Start from player after big blind
    game.current_player_idx = 1

    done = False
    while not done:
        player = game.players[game.current_player_idx]
        to_call = game.current_bet - player.current_bet
        action = player.decide_action(to_call, game.community_cards)
        _, _, done, _ = game.step(action)

    assert ai1.all_in is True
    assert game.pot >= 20

def test_bug_fold_removes_player_from_active():
    p1 = Player("P1", stack=1000, is_human=False)
    p2 = Player("P2", stack=1000, is_human=False)

    game = PokerGame([p1, p2], game_mode=GameMode.AI_VS_AI)
    game.reset_for_new_hand()
    game.dealer_position = 0
    game.post_blinds()
    game.deal_hole_cards()

    p1.fold()
    if p1 in game.active_players:
        game.active_players.remove(p1)

    assert p1 not in game.active_players
    assert p2 in game.active_players

def test_multi_hand_session():
    random.seed(123)
    players = [
        Player("Alice", stack=1000, is_human=False),
        Player("Bob", stack=1000, is_human=False),
        Player("Carol", stack=1000, is_human=False)
    ]

    game = PokerGame(players, game_mode=GameMode.AI_VS_AI)
    num_hands = 5

    for hand_num in range(num_hands):
        # Suppress print for cleaner test output
        import builtins
        orig_print = builtins.print
        builtins.print = lambda *args, **kwargs: None

        initial_total = sum(p.stack for p in players)

        game.play_hand()

        final_total = sum(p.stack for p in players)
        assert final_total == initial_total, "Total chips should remain constant"

        for p in players:
            assert p.stack >= 0, f"{p.name} has negative stack"

        expected_dealer = (hand_num + 1) % len(players)
        assert game.dealer_position == expected_dealer, "Dealer position mismatch"

        builtins.print = orig_print
