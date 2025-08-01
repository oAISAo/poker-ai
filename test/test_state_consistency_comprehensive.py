#!/usr/bin/env python3
"""
Comprehensive tests for state consistency validation and fixing methods.
These tests target the new state validation code to improve test coverage.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from engine.game import PokerGame
from engine.player import Player
from env.multi_table_tournament_env import MultiTableTournamentEnv

class TestStateConsistencyValidation:
    """Test suite for state consistency validation methods"""
    
    def test_validate_state_consistency_normal_state(self):
        """Test that normal game state validates correctly"""
        players = [Player(f"Player{i}", stack=1000) for i in range(3)]
        game = PokerGame(players, small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)
        
        # Normal state should validate
        assert game._validate_state_consistency("normal state test")
    
    def test_inconsistency_player_bet_exceeds_game_bet_after_blinds(self):
        """Reproduce and debug player.current_bet > game.current_bet after posting blinds (as seen in Sharky 1.0.1 training)"""
        players = [Player(f"Player{i}", stack=1000) for i in range(3)]
        game = PokerGame(players, small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)

        # Simulate posting blinds
        game.players[0].current_bet = 50  # SB posts too much
        game.players[1].current_bet = 20  # BB normal
        game.current_bet = 0  # Incorrect game bet (should be max player bet)

        print('[DEBUG] Before fix:', {
            'player0.current_bet': game.players[0].current_bet,
            'player1.current_bet': game.players[1].current_bet,
            'game.current_bet': game.current_bet,
            'pot': game.pot
        })

        # Should detect inconsistency
        assert not game._validate_state_consistency('after posting blinds')

        # Fix the state
        assert game.fix_state_inconsistencies()

        print('[DEBUG] After fix:', {
            'player0.current_bet': game.players[0].current_bet,
            'player1.current_bet': game.players[1].current_bet,
            'game.current_bet': game.current_bet,
            'pot': game.pot
        })

        # State should now be consistent
        assert game._validate_state_consistency('after fixing blinds inconsistency')
    
    def test_validate_state_consistency_player_bet_exceeds_game_bet(self):
        """Test detection of player.current_bet > game.current_bet"""
        players = [Player(f"Player{i}", stack=1000) for i in range(3)]
        game = PokerGame(players, small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)
        
        # Create artificial inconsistency
        game.players[0].current_bet = game.current_bet + 100
        
        # Should detect inconsistency
        assert not game._validate_state_consistency("player bet exceeds game bet")
    
    def test_validate_state_consistency_game_bet_mismatch(self):
        """Test detection of game.current_bet not matching max player bet"""
        players = [Player(f"Player{i}", stack=1000) for i in range(3)]
        game = PokerGame(players, small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)
        
        # Set game.current_bet lower than max player bet
        max_player_bet = max(p.current_bet for p in game.players)
        game.current_bet = max_player_bet - 5
        
        # Should detect inconsistency
        assert not game._validate_state_consistency("game bet mismatch")
    
    def test_synchronize_current_bet(self):
        """Test synchronization of game.current_bet with max player bet"""
        players = [Player(f"Player{i}", stack=1000) for i in range(3)]
        game = PokerGame(players, small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)
        
        # Manually create inconsistency
        game.players[1].current_bet = 50  # Higher than game.current_bet
        original_game_bet = game.current_bet
        
        # Synchronize
        game._synchronize_current_bet()
        
        # Should synchronize to max player bet
        assert game.current_bet == 50
        assert game.current_bet != original_game_bet
    
    def test_fix_state_inconsistencies_success(self):
        """Test successful fixing of state inconsistencies"""
        players = [Player(f"Player{i}", stack=1000) for i in range(3)]
        game = PokerGame(players, small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)
        
        # Create fixable inconsistency (game.current_bet lower than max player bet)
        game.players[0].current_bet = 100
        game.current_bet = 50  # Lower than max player bet
        
        # Fix should succeed
        assert game.fix_state_inconsistencies()
        assert game._validate_state_consistency("after fix")
    
    def test_fix_state_inconsistencies_player_bet_too_high(self):
        """Test fixing when player.current_bet > game.current_bet"""
        players = [Player(f"Player{i}", stack=1000) for i in range(3)]
        game = PokerGame(players, small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)
        
        # Create inconsistency where player bet exceeds game bet
        original_game_bet = game.current_bet
        original_stack = game.players[0].stack
        game.players[0].current_bet = original_game_bet + 50
        
        # Fix should reduce player bet and refund excess to stack
        assert game.fix_state_inconsistencies()
        assert game.players[0].current_bet == original_game_bet
        assert game.players[0].stack == original_stack + 50  # Excess refunded
    
    def test_state_validation_in_step_method(self):
        """Test that state validation is called during step method"""
        players = [Player(f"Player{i}", stack=1000) for i in range(3)]
        game = PokerGame(players, small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)
        
        # Execute a step and verify no validation errors
        game.current_player_idx = 0
        to_call = game.current_bet - game.players[0].current_bet
        
        if to_call > 0:
            game.step("call", 0)
        else:
            game.step("check", 0)
        
        # State should remain consistent
        assert game._validate_state_consistency("after step")
    
    def test_state_validation_after_raise(self):
        """Test that state remains consistent after raise operations"""
        players = [Player(f"Player{i}", stack=1000) for i in range(3)]
        game = PokerGame(players, small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)
        
        # Execute a raise
        game.current_player_idx = 0
        game.step("raise", 50)
        
        # State should be consistent after raise
        assert game._validate_state_consistency("after raise")
        assert game.current_bet == 50
        assert game.players[0].current_bet == 50

class TestTournamentStateConsistency:
    """Test state consistency in tournament environment"""
    
    def test_tournament_state_validation_on_step(self):
        """Test that tournament validates state on each step"""
        env = MultiTableTournamentEnv(total_players=6, max_players_per_table=3)
        obs, info = env.reset()
        
        # Execute several actions and verify state consistency
        for _ in range(10):
            action_mask = info.get("action_mask", [True, True, True])
            if any(action_mask):
                # Choose first legal action
                action = next(i for i, legal in enumerate(action_mask) if legal)
                obs, reward, done, truncated, info = env.step(action)
                
                if done:
                    break
    
    def test_tournament_state_fixing_on_inconsistency(self):
        """Test that tournament can fix state inconsistencies"""
        env = MultiTableTournamentEnv(total_players=6, max_players_per_table=3)
        obs, info = env.reset()
        
        # Get active table and artificially create inconsistency
        table = env.tables[env.active_table_id]
        if table.players:
            # Create artificial inconsistency
            table.game.players[0].current_bet = table.game.current_bet + 100
            
            # Tournament should detect and fix this on next step
            action_mask = info.get("action_mask", [True, True, True])
            if any(action_mask):
                action = next(i for i, legal in enumerate(action_mask) if legal)
                # This should trigger state validation and fixing
                obs, reward, done, truncated, info = env.step(action)
    
    def test_pot_mismatch_detection_and_fixing(self):
        """Test pot mismatch detection in showdown"""
        players = [Player(f"Player{i}", stack=1000) for i in range(2)]
        game = PokerGame(players, small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)
        
        # Simulate a hand to showdown
        game.current_player_idx = 0
        game.step("call", 0)  # Call
        game.current_player_idx = 1
        game.step("check", 0)  # Check
        
        # Force to showdown
        game.phase_idx = game.PHASES.index("showdown")
        
        # Artificially create pot mismatch
        original_pot = game.pot
        game.pot = original_pot + 10  # Create mismatch
        
        # Showdown should detect and potentially fix mismatch
        game.showdown()
    
    def test_reproduce_sharky_training_state_inconsistency(self):
        """Reproduce the state inconsistency seen in Sharky 1.0.1 training: player.current_bet > game.current_bet, game.current_bet != max player bet, total player bets != game.pot"""
        env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9)
        obs, info = env.reset()

        # Pick a table and player to simulate the inconsistency
        table = env.tables[env.active_table_id]
        game = table.game
        # Simulate posting blinds and new hand setup
        game.reset_for_new_hand(is_first_hand=True)

        # Artificially set player and game bets to reproduce the warning
        game.players[1].current_bet = 50  # e.g. TAG_1
        game.current_bet = 0
        game.pot = 130  # Arbitrary pot value to mismatch

        total_player_bets = sum(p.current_bet for p in game.players)
        max_player_bet = max(p.current_bet for p in game.players)

        print('[DEBUG] Before fix:', {
            'player1.current_bet': game.players[1].current_bet,
            'game.current_bet': game.current_bet,
            'max_player_bet': max_player_bet,
            'total_player_bets': total_player_bets,
            'game.pot': game.pot
        })

        # Assert all warning conditions
        assert game.players[1].current_bet > game.current_bet
        assert game.current_bet != max_player_bet
        assert total_player_bets != game.pot

        # Fix the state
        assert game.fix_state_inconsistencies()

        total_player_bets_after = sum(p.current_bet for p in game.players)
        max_player_bet_after = max(p.current_bet for p in game.players)

        print('[DEBUG] After fix:', {
            'player1.current_bet': game.players[1].current_bet,
            'game.current_bet': game.current_bet,
            'max_player_bet': max_player_bet_after,
            'total_player_bets': total_player_bets_after,
            'game.pot': game.pot
        })

        # Assert state is now consistent
        assert game.players[1].current_bet <= game.current_bet
        assert game.current_bet == max_player_bet_after
        assert total_player_bets_after == game.pot

class TestAdvancedStateScenarios:
    """Test complex state scenarios"""
    
    def test_state_consistency_with_all_in_players(self):
        """Test state consistency with all-in players"""
        alice = Player("Alice", stack=50)
        bob = Player("Bob", stack=1000)
        players = [alice, bob]
        game = PokerGame(players, small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)
        
        # Alice goes all-in
        game.current_player_idx = 0  # Alice (SB in heads-up)
        game.step("raise", alice.stack + alice.current_bet)
        
        # State should be consistent
        assert game._validate_state_consistency("after all-in")
        assert alice.all_in
        assert alice.stack == 0
    
    def test_state_consistency_during_side_pots(self):
        """Test state consistency in side pot scenarios"""
        alice = Player("Alice", stack=1000)
        bob = Player("Bob", stack=100)
        charlie = Player("Charlie", stack=1000)
        
        game = PokerGame([alice, bob, charlie], small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)
        
        # Create side pot scenario
        game.current_player_idx = 0  # Alice
        game.step("raise", 200)  # More than Bob can match
        
        game.current_player_idx = 1  # Bob
        game.step("call", 0)  # Bob goes all-in
        
        # State should remain consistent despite different bet amounts
        assert game._validate_state_consistency("during side pot creation")
    
    def test_state_validation_edge_cases(self):
        """Test state validation with edge cases"""
        players = [Player(f"Player{i}", stack=1000) for i in range(3)]
        game = PokerGame(players, small_blind=10, big_blind=20)
        game.reset_for_new_hand(is_first_hand=True)
        
        # Test with zero current_bet
        game.current_bet = 0
        for player in game.players:
            player.current_bet = 0
        
        assert game._validate_state_consistency("zero bets")
        
        # Test with all players having same bet
        game.current_bet = 50
        for player in game.players:
            player.current_bet = 50
        
        assert game._validate_state_consistency("all equal bets")
    
    def test_table_balancing_consistency(self):
        """Simulate a multi-table tournament with eliminations and table balancing, checking for state consistency after each balancing event."""
        env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9)
        obs, info = env.reset()

        # Simulate eliminations to trigger table balancing
        # We'll manually eliminate players from table 0
        table0 = env.tables[0]
        eliminated = 0
        for player in list(table0.players):
            if eliminated < 5:  # Eliminate 5 players to force balancing
                player.stack = 0
                player.in_hand = False
                eliminated += 1

        # Trigger table balancing (simulate tournament step)
        env.balance_table(0)

        # [DEBUG] Print state after balancing
        for tid, table in env.tables.items():
            print(f'[DEBUG] Table {tid} after balancing:')
            for player in table.players:
                print(f'    {player.name}.current_bet = {player.current_bet}, stack = {player.stack}, in_hand = {player.in_hand}')
            print(f'    table.game.current_bet = {table.game.current_bet}')
            print(f'    table.game.pot = {table.game.pot}')

            # Assert state consistency for each table
            assert table.game._validate_state_consistency(f'table {tid} after balancing')

if __name__ == "__main__":
    pytest.main(["-v", __file__])
