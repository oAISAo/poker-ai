#!/usr/bin/env python3

"""
Debug script to identify pot mismatch calculations in side pot scenarios.
This will help us find the specific issues with side pot logic.
"""

from engine.game import PokerGame
from engine.player import Player

def test_side_pot_scenario_1():
    """Test basic side pot scenario with all-in player"""
    print("\n=== Test 1: Basic Side Pot Scenario ===")
    
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=100)    # Short stack
    charlie = Player("Charlie", stack=1000)
    
    game = PokerGame([alice, bob, charlie], small_blind=10, big_blind=20)
    game.reset_for_new_hand(is_first_hand=True)
    
    print(f"[DEBUG] Initial pot after blinds: {game.pot}")
    print(f"[DEBUG] Player contributions after blinds:")
    for p in game.players:
        print(f"  {p.name}: total_contributed={p.total_contributed}, current_bet={p.current_bet}, stack={p.stack}")
    
    # Alice raises to 300 (more than Bob can match)
    game.current_player_idx = 0
    game.step("raise", 300)
    
    print(f"[DEBUG] After Alice raises to 300:")
    print(f"  Pot: {game.pot}")
    for p in game.players:
        print(f"  {p.name}: total_contributed={p.total_contributed}, current_bet={p.current_bet}, stack={p.stack}")
    
    # Bob goes all-in for 100
    game.current_player_idx = 1
    game.step("call", 0)  # Bob goes all-in
    
    print(f"[DEBUG] After Bob goes all-in:")
    print(f"  Pot: {game.pot}")
    for p in game.players:
        print(f"  {p.name}: total_contributed={p.total_contributed}, current_bet={p.current_bet}, stack={p.stack}")
    
    # Charlie calls 300
    game.current_player_idx = 2
    game.step("call", 0)
    
    print(f"[DEBUG] After Charlie calls:")
    print(f"  Pot: {game.pot}")
    for p in game.players:
        print(f"  {p.name}: total_contributed={p.total_contributed}, current_bet={p.current_bet}, stack={p.stack}")
    
    # Force to showdown to see side pot calculation
    game.phase_idx = game.PHASES.index("showdown")
    print(f"\n[DEBUG] Forcing showdown to check side pot calculation...")
    game.showdown()

def test_side_pot_scenario_2():
    """Test multiple all-in scenario with different contribution levels"""
    print("\n=== Test 2: Multiple All-In Scenario ===")
    
    alice = Player("Alice", stack=50)    # Shortest stack
    bob = Player("Bob", stack=200)       # Medium stack
    charlie = Player("Charlie", stack=1000)  # Large stack
    
    game = PokerGame([alice, bob, charlie], small_blind=10, big_blind=20)
    game.reset_for_new_hand(is_first_hand=True)
    
    print(f"[DEBUG] Initial pot after blinds: {game.pot}")
    
    # Alice goes all-in (she only has 50, so she can't raise to 500)
    game.current_player_idx = 0  # Alice
    game.step("call", 0)  # Alice calls the BB for her remaining 30 chips (50-20=30 to call)
    
    game.current_player_idx = 1  # Bob
    game.step("raise", 200)  # Bob raises to 200
    
    game.current_player_idx = 2  # Charlie
    game.step("call", 0)  # Charlie calls Bob's raise
    
    print(f"[DEBUG] After all actions:")
    print(f"  Pot: {game.pot}")
    for p in game.players:
        print(f"  {p.name}: total_contributed={p.total_contributed}, current_bet={p.current_bet}, stack={p.stack}")
    
    # Force to showdown
    game.phase_idx = game.PHASES.index("showdown")
    print(f"\n[DEBUG] Forcing showdown to check side pot calculation...")
    game.showdown()

def test_side_pot_scenario_3():
    """Test scenario where blinds/antes affect side pot calculation"""
    print("\n=== Test 3: Blinds/Antes Side Pot Scenario ===")
    
    alice = Player("Alice", stack=1000)
    bob = Player("Bob", stack=25)    # Very short stack (less than BB)
    charlie = Player("Charlie", stack=1000)
    
    game = PokerGame([alice, bob, charlie], small_blind=10, big_blind=20, ante=5)
    game.reset_for_new_hand(is_first_hand=True)
    
    print(f"[DEBUG] Initial pot after blinds/antes: {game.pot}")
    print(f"[DEBUG] Player states after blinds:")
    for p in game.players:
        print(f"  {p.name}: total_contributed={p.total_contributed}, current_bet={p.current_bet}, stack={p.stack}")
    
    # Alice raises
    game.current_player_idx = 0
    game.step("raise", 100)
    
    # Bob already all-in from blinds (if stack < BB)
    if bob.stack > 0:
        game.current_player_idx = 1
        game.step("call", 0)
    
    # Charlie calls or checks
    game.current_player_idx = 2
    to_call_charlie = game.current_bet - charlie.current_bet
    if to_call_charlie > 0:
        game.step("call", 0)
    else:
        game.step("check", 0)
    
    print(f"[DEBUG] After all actions:")
    print(f"  Pot: {game.pot}")
    for p in game.players:
        print(f"  {p.name}: total_contributed={p.total_contributed}, current_bet={p.current_bet}, stack={p.stack}")
    
    # Force to showdown
    game.phase_idx = game.PHASES.index("showdown")
    print(f"\n[DEBUG] Forcing showdown to check side pot calculation...")
    game.showdown()

if __name__ == "__main__":
    print("=== Debugging Pot Mismatch Calculations ===")
    
    try:
        test_side_pot_scenario_1()
        test_side_pot_scenario_2()
        test_side_pot_scenario_3()
        
        print("\n=== Debug Complete ===")
        print("Look for [WARNING] Pot mismatch messages above to identify issues.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
