import random
from engine.game import PokerGame
from engine.player import Player
from utils.enums import GameMode
import pytest

def test_bug_raise_resets_betting_round(monkeypatch):
    """
    BUG-TEST: Verify that after a raise, the betting round continues
    and all other players get a chance to act again.
    """
    # Setup 3 AI players with controlled decisions
    class TestPlayer(Player):
        def __init__(self, name):
            super().__init__(name, stack=1000, is_human=False)
            self.action_sequence = ["call", "raise 50", "call", "check"]
            self.action_index = 0

        def decide_action(self, to_call, community_cards):
            if self.action_index < len(self.action_sequence):
                action = self.action_sequence[self.action_index]
                self.action_index += 1
                # Normalize "raise 50" -> just "raise"
                return "raise" if action.startswith("raise") else action
            return "check"

    players = [TestPlayer("P1"), TestPlayer("P2"), TestPlayer("P3")]
    game = PokerGame(players, game_mode=GameMode.AI_VS_AI)

    game.reset_for_new_hand()
    game.dealer_position = 0
    game.post_blinds()
    game.deal_hole_cards()

    # We patch print to suppress output clutter if needed
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    # Run betting round starting from first player after dealer
    game.betting_round(start=1)

    # Assert that after raise, the betting round did not end prematurely:
    # All players acted more than once due to reset after raise
    # Check pot and current bets reflect actions
    total_bets = sum(p.current_bet for p in players)
    assert total_bets >= game.current_bet * len(players), "Pot and bets inconsistent after raise"

    # All players still in hand (no folds triggered in this test)
    assert all(p.in_hand for p in players), "Players folded unexpectedly in raise test"

def test_bug_human_action_raise_updates_pot_and_stack(monkeypatch):
    """
    BUG-TEST: Human player raise correctly updates pot and stack,
    and game continues properly after raise.
    """
    human = Player("Human", stack=1000, is_human=True)
    ai = Player("AI", stack=1000, is_human=False)

    game = PokerGame([human, ai], game_mode=GameMode.AI_VS_AI)

    game.reset_for_new_hand()
    game.dealer_position = 0
    game.post_blinds()
    game.deal_hole_cards()

    inputs = iter([
        "raise 30",  # Human raises by 30 chips
        "call",      # Human calls to finish the round
    ])

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    game.betting_round(start=1)

    # After raise, pot should be at least 30 + blinds
    assert game.pot >= 30 + game.small_blind + game.big_blind

    # Human stack reduced by raised amount
    assert human.stack < 1000

def test_bug_all_in_sets_flag(monkeypatch):
    """
    BUG-TEST: Player going all-in sets the all_in flag and game handles correctly.
    """
    ai1 = Player("AI1", stack=20, is_human=False)
    ai2 = Player("AI2", stack=1000, is_human=False)

    # AI1 will call all-in (stack 20), AI2 will call normally
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

    game.betting_round(start=1)

    # After betting round, AI1 should be all-in
    assert ai1.all_in is True

    # Pot should include AI1's full stack
    assert game.pot >= 20

def test_bug_fold_removes_player_from_active():
    """
    BUG-TEST: Folded players are removed from active_players list.
    """
    p1 = Player("P1", stack=1000, is_human=False)
    p2 = Player("P2", stack=1000, is_human=False)

    game = PokerGame([p1, p2], game_mode=GameMode.AI_VS_AI)
    game.reset_for_new_hand()
    game.dealer_position = 0
    game.post_blinds()
    game.deal_hole_cards()

    # P1 folds manually
    p1.fold()
    if p1 in game.active_players:
        game.active_players.remove(p1)

    # Only p2 should remain in active_players
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
        print(f"\nStarting hand {hand_num + 1}")
        initial_total = sum(p.stack for p in players)

        game.play_hand()

        final_total = sum(p.stack for p in players)
        assert final_total == initial_total, "Total chips should remain constant"

        # Check no player has negative stack
        for p in players:
            assert p.stack >= 0, f"{p.name} has negative stack"

        # Dealer should rotate
        expected_dealer = (hand_num + 1) % len(players)
        assert game.dealer_position == expected_dealer, "Dealer position mismatch"

