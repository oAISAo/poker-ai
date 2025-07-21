import pytest
from engine.game import PokerGame
from engine.player import Player
from utils.enums import GameMode

def test_basic_game_flow():
    # Setup two players
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=1000)

    # Create game instance (AI vs AI mode)
    game = PokerGame([alice, bob], game_mode=GameMode.AI_VS_AI)

    # Play a single hand
    game.play_hand()

    # Total stack must be conserved (ignoring rake or errors)
    total_stack_after = alice.stack + bob.stack
    assert total_stack_after == 2000

    # At least one player should have changed stack
    assert alice.stack != 1000 or bob.stack != 1000

    # Players must not have negative chips
    assert alice.stack >= 0
    assert bob.stack >= 0

    # Players' hole cards should be cleared
    assert alice.hole_cards == []
    assert bob.hole_cards == []

def test_player_fold_behavior():
    player = Player("TestPlayer", stack=1000)
    player.fold()
    assert player.in_hand is False

def test_player_bet_and_all_in():
    player = Player("TestPlayer", stack=50)
    bet = player.bet_chips(100)
    # Bet cannot exceed stack, so should be 50
    assert bet == 50
    assert player.all_in is True
    assert player.stack == 0

def test_player_reset_for_new_hand():
    player = Player("TestPlayer", stack=1000)
    player.bet_chips(100)
    player.fold()
    player.reset_for_new_hand()
    assert player.in_hand is True
    assert player.current_bet == 0
    assert player.all_in is False
    assert player.hole_cards == []

if __name__ == "__main__":
    pytest.main(["-v", __file__])
