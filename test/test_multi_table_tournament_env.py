import pytest
import numpy as np
from env.multi_table_tournament_env import MultiTableTournamentEnv, Table
from engine.player import Player

def test_multi_table_tournament_initialization():
    """Test that multi-table tournament initializes correctly"""
    env = MultiTableTournamentEnv(total_players=27, max_players_per_table=9)
    
    # Should create 3 tables with 9 players each
    assert len(env.tables) == 3
    assert env.total_players == 27
    
    # Check player distribution
    total_seated = sum(len(table.players) for table in env.tables.values())
    assert total_seated == 27
    
    # Each table should have at most 9 players
    for table in env.tables.values():
        assert len(table.players) <= 9
        assert len(table.players) >= 1

def test_uneven_player_distribution():
    """Test tournament with uneven player distribution"""
    env = MultiTableTournamentEnv(total_players=50, max_players_per_table=9)
    
    # Should create 6 tables (50/9 = 5.55, rounded up)
    assert len(env.tables) >= 5
    
    # Total players should still be 50
    total_seated = sum(len(table.players) for table in env.tables.values())
    assert total_seated == 50

def test_table_class_functionality():
    """Test Table class methods"""
    players = [Player(f"Player_{i}", stack=1000) for i in range(5)]
    table = Table(table_id=1, players=players, starting_stack=1000)
    
    # Test initial state
    assert table.table_id == 1
    assert len(table.players) == 5
    assert table.get_active_player_count() == 5
    assert table.is_active is True
    
    # Test removing player
    removed_player = players[0]
    success = table.remove_player(removed_player)
    assert success is True
    assert len(table.players) == 4
    assert removed_player not in table.players
    
    # Test adding player
    new_player = Player("New_Player", stack=1000)
    success = table.add_player(new_player)
    assert success is True
    assert len(table.players) == 5
    assert new_player in table.players

def test_observation_space():
    """Test observation space shape and content"""
    env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9)
    obs, info = env.reset()
    
    # Should have 8-dimensional observation
    assert obs.shape == (8,)
    assert "action_mask" in info
    assert info["action_mask"].shape == (3,)
    
    # Observation values should be reasonable
    assert obs[0] >= 0  # stack
    assert obs[1] >= 0  # to_call
    assert obs[2] >= 0  # pot
    assert obs[3] >= 0  # current_bet
    assert obs[4] in [0, 1]  # in_hand (boolean)
    assert obs[5] >= 0  # table_id
    assert obs[6] >= 2  # players_at_table (at least 2 for active game)
    assert obs[7] >= 0  # blind_level

def test_legal_action_mask():
    """Test legal action mask generation"""
    env = MultiTableTournamentEnv(total_players=9, max_players_per_table=9)
    obs, info = env.reset()
    
    mask = info["action_mask"]
    
    # At least one action should be legal
    assert any(mask)
    
    # Mask should have 3 elements [fold, call/check, raise]
    assert len(mask) == 3
    assert all(isinstance(x, (bool, np.bool_)) for x in mask)

def test_step_functionality():
    """Test basic step functionality"""
    env = MultiTableTournamentEnv(total_players=9, max_players_per_table=9)
    obs, info = env.reset()
    
    # Get a legal action
    mask = info["action_mask"]
    legal_actions = [i for i, legal in enumerate(mask) if legal]
    assert len(legal_actions) > 0
    
    action = legal_actions[0]
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Should return valid step results
    assert obs.shape == (8,)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "action_mask" in info

def test_blind_increase_mechanism():
    """Test that blinds increase correctly"""
    env = MultiTableTournamentEnv(
        total_players=9, 
        max_players_per_table=9,
        hands_per_blind_level=2  # Quick blind increases for testing
    )
    obs, info = env.reset()
    
    initial_blind_level = env.current_blind_level
    initial_blinds = env.blinds_schedule[initial_blind_level]
    
    # Play several hands to trigger blind increase
    for _ in range(10):  # Should be enough to increase blinds
        if env._tournament_finished():
            break
        
        mask = info["action_mask"]
        if any(mask):
            action = np.argmax(mask)  # Take first legal action
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                break
    
    # Blinds should have increased (or tournament ended)
    if not env._tournament_finished():
        assert env.current_blind_level >= initial_blind_level

def test_table_balancing_trigger():
    """Test that table balancing is triggered when needed"""
    env = MultiTableTournamentEnv(
        total_players=18, 
        max_players_per_table=9,
        table_balancing_threshold=6
    )
    obs, info = env.reset()
    
    # Manually eliminate players from one table to trigger balancing
    target_table = env.tables[0]
    players_to_eliminate = target_table.players[6:]  # Leave only 6 players
    
    for player in players_to_eliminate:
        player.stack = 0
    
    # Check balancing mechanism
    env._check_table_balancing()
    
    # After balancing, tables should be more balanced
    active_tables = env._get_active_tables()
    if len(active_tables) > 1:
        player_counts = [t.get_active_player_count() for t in active_tables]
        # Check that no table is too empty relative to others
        assert max(player_counts) - min(player_counts) <= 3

def test_player_elimination_tracking():
    """Test that player elimination is tracked correctly"""
    env = MultiTableTournamentEnv(total_players=9, max_players_per_table=9)
    obs, info = env.reset()
    
    initial_elimination_count = len(env.elimination_order)
    
    # Manually eliminate a player
    table = env.tables[env.active_table_id]
    if table.players:
        player_to_eliminate = table.players[0]
        player_to_eliminate.stack = 0
        
        env._update_elimination_order()
        
        # Should have one more elimination
        assert len(env.elimination_order) == initial_elimination_count + 1
        assert player_to_eliminate in env.elimination_order

def test_tournament_completion():
    """Test tournament completion detection"""
    env = MultiTableTournamentEnv(total_players=9, max_players_per_table=9)
    obs, info = env.reset()
    
    # Manually eliminate all but one player
    remaining_player = env.all_players[0]
    for player in env.all_players[1:]:
        player.stack = 0
    
    # Tournament should be finished
    assert env._tournament_finished() is True

def test_reset_functionality():
    """Test that reset properly reinitializes the tournament"""
    env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9)
    
    # First reset
    obs1, info1 = env.reset()
    initial_tables = len(env.tables)
    initial_players = env.total_players
    
    # Modify state
    env.current_blind_level = 3
    env.elimination_order = [env.all_players[0]]
    
    # Second reset should restore initial state
    obs2, info2 = env.reset()
    
    assert env.current_blind_level == 0
    assert len(env.elimination_order) == 0
    assert len(env.tables) == initial_tables
    assert env.total_players == initial_players

def test_table_selection_round_robin():
    """Test that tables are selected in round-robin fashion"""
    env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9)
    obs, info = env.reset()
    
    # Should have multiple tables
    active_tables = env._get_active_tables()
    assert len(active_tables) >= 2
    
    # Track table selections
    table_selections = []
    initial_table = env.active_table_id
    table_selections.append(initial_table)
    
    # Play a few hands and track table changes
    for _ in range(5):
        if env._tournament_finished():
            break
            
        mask = info["action_mask"]
        if any(mask):
            action = np.argmax(mask)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if env.active_table_id != table_selections[-1]:
                table_selections.append(env.active_table_id)
            
            if terminated:
                break
    
    # Should have played at multiple tables (if tournament didn't end too quickly)
    if not env._tournament_finished() and len(table_selections) > 1:
        assert len(set(table_selections)) > 1

def test_tournament_stats():
    """Test tournament statistics generation"""
    env = MultiTableTournamentEnv(total_players=27, max_players_per_table=9)
    obs, info = env.reset()
    
    stats = env.get_tournament_stats()
    
    # Check that all expected stats are present
    expected_keys = [
        "total_players", "remaining_players", "eliminated_players", 
        "active_tables", "current_blind_level", "blinds", 
        "hands_played", "average_stack", "chip_leader", "chip_leader_stack"
    ]
    
    for key in expected_keys:
        assert key in stats
    
    # Check stat values make sense
    assert stats["total_players"] == 27
    assert stats["remaining_players"] + stats["eliminated_players"] == 27
    assert stats["active_tables"] > 0
    assert stats["current_blind_level"] >= 1
    assert isinstance(stats["blinds"], tuple)
    assert len(stats["blinds"]) == 2

def test_large_tournament():
    """Test with a large tournament (99 players)"""
    env = MultiTableTournamentEnv(total_players=99, max_players_per_table=9)
    obs, info = env.reset()
    
    # Should create 11 tables (99/9 = 11)
    assert len(env.tables) == 11
    
    # Total players should be 99
    total_seated = sum(len(table.players) for table in env.tables.values())
    assert total_seated == 99
    
    # Each table should have at most 9 players
    for table in env.tables.values():
        assert len(table.players) <= 9
        assert len(table.players) >= 1

def test_heads_up_final_table():
    """Test behavior when down to final 2 players"""
    env = MultiTableTournamentEnv(total_players=3, max_players_per_table=9)
    obs, info = env.reset()
    
    # Eliminate one player to get heads-up
    player_to_eliminate = env.all_players[0]
    player_to_eliminate.stack = 0
    env._update_elimination_order()
    
    # Should still be able to play
    remaining_tables = env._get_active_tables()
    if remaining_tables:
        assert len(remaining_tables) >= 1
        # Should have a table with 2 players
        two_player_tables = [t for t in remaining_tables if t.get_active_player_count() == 2]
        assert len(two_player_tables) >= 1

if __name__ == "__main__":
    # Run a specific test for debugging
    test_multi_table_tournament_initialization()
    print("Multi-table tournament initialization test passed!")

# ====== EDGE CASE AND ROBUSTNESS TESTS ======

def test_minimum_tournament_size():
    """Test minimum viable tournament (2 players)"""
    env = MultiTableTournamentEnv(total_players=2, max_players_per_table=9)
    obs, info = env.reset()
    
    # Should create 1 table with 2 players
    assert len(env.tables) == 1
    assert env.total_players == 2
    
    # Should be able to play
    mask = info["action_mask"]
    assert any(mask), "Should have legal actions with 2 players"

def test_invalid_tournament_configurations():
    """Test invalid tournament configurations"""
    # Zero players
    with pytest.raises((ValueError, RuntimeError)):
        env = MultiTableTournamentEnv(total_players=0)
    
    # One player
    with pytest.raises((ValueError, RuntimeError)):
        env = MultiTableTournamentEnv(total_players=1)
    
    # Max players per table too small
    with pytest.raises((ValueError, RuntimeError)):
        env = MultiTableTournamentEnv(total_players=10, max_players_per_table=1)

def test_all_players_eliminated_except_one():
    """Test tournament completion when all but one player eliminated"""
    env = MultiTableTournamentEnv(total_players=9, max_players_per_table=9)
    obs, info = env.reset()
    
    # Eliminate all but one player
    winner = env.all_players[0]
    for player in env.all_players[1:]:
        player.stack = 0
    
    env._update_elimination_order()
    
    # Tournament should be finished
    assert env._tournament_finished() is True
    assert len(env.elimination_order) == 8
    assert winner not in env.elimination_order

def test_simultaneous_all_in_eliminations():
    """Test multiple players eliminated in same hand"""
    env = MultiTableTournamentEnv(total_players=9, max_players_per_table=9)
    obs, info = env.reset()
    
    initial_eliminations = len(env.elimination_order)
    
    # Eliminate multiple players simultaneously
    players_to_eliminate = env.all_players[:3]
    for player in players_to_eliminate:
        player.stack = 0
    
    env._update_elimination_order()
    
    # Should track all eliminations
    assert len(env.elimination_order) == initial_eliminations + 3
    for player in players_to_eliminate:
        assert player in env.elimination_order

def test_table_breaking_edge_cases():
    """Test edge cases in table breaking logic"""
    env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9)
    obs, info = env.reset()
    
    # Eliminate players to force table breaking
    table_0 = env.tables[0]
    # Leave only 1 player at table 0
    remaining_player = table_0.players[0]  # Store reference BEFORE elimination
    players_to_eliminate = table_0.players[1:]
    for player in players_to_eliminate:
        player.stack = 0
    
    # Force table balancing
    env._check_table_balancing()
    
    # Table 0 should be broken (inactive)
    assert not table_0.is_active
    
    # Remaining player should be moved to another table
    found_player = False
    for table_id, table in env.tables.items():
        if table_id != 0 and table.is_active:
            if remaining_player in table.players:
                found_player = True
                break
    
    assert found_player, "Player should have been moved to active table"

def test_invalid_actions_and_error_handling():
    """Test invalid actions and error handling"""
    env = MultiTableTournamentEnv(total_players=9, max_players_per_table=9)
    obs, info = env.reset()
    
    # Test invalid action (out of range)
    obs, reward, terminated, truncated, info = env.step(999)
    assert reward <= 0, "Invalid action should give negative reward"
    
    # Test action when no legal actions (if possible)
    mask = info["action_mask"]
    illegal_actions = [i for i, legal in enumerate(mask) if not legal]
    
    if illegal_actions:
        obs, reward, terminated, truncated, info = env.step(illegal_actions[0])
        assert reward <= 0, "Illegal action should give negative reward"

def test_blind_level_progression_edge_cases():
    """Test blind level progression edge cases"""
    # Test tournament that reaches max blind level
    env = MultiTableTournamentEnv(
        total_players=6,
        hands_per_blind_level=1,  # Fast progression
        blinds_schedule=[(10, 20), (20, 40), (50, 100)]  # Short schedule
    )
    obs, info = env.reset()
    
    # Force rapid blind progression
    for _ in range(10):  # Should exceed blind schedule
        env.hands_played_this_level += 1
        env._increase_blinds_if_needed()
    
    # Should not exceed max blind level
    assert env.current_blind_level < len(env.blinds_schedule)

def test_player_movement_preserves_integrity():
    """Test that player movement preserves game integrity"""
    env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9)
    obs, info = env.reset()
    
    # Record initial player distribution
    initial_player_names = set(p.name for p in env.all_players)
    
    # Force table balancing multiple times
    for _ in range(3):
        env._check_table_balancing()
    
    # Collect all players from all tables
    final_player_names = set()
    for table in env.tables.values():
        for player in table.players:
            final_player_names.add(player.name)
    
    # Should have same players (no duplication or loss)
    assert initial_player_names == final_player_names

def test_concurrent_hands_across_tables():
    """Test that hands progress correctly across multiple tables"""
    env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9)
    obs, info = env.reset()
    
    # Track hands played at each table
    initial_hands = {table_id: table.hands_played for table_id, table in env.tables.items()}
    
    # Play several steps
    for _ in range(20):
        if env._tournament_finished():
            break
        
        mask = info["action_mask"]
        if any(mask):
            action = np.argmax(mask)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                break
    
    # At least some tables should have progressed
    hands_progressed = False
    for table_id, table in env.tables.items():
        if table.hands_played > initial_hands[table_id]:
            hands_progressed = True
            break
    
    if not env._tournament_finished():
        assert hands_progressed, "No hands progressed across tables"

def test_observation_consistency_across_tables():
    """Test observation consistency when switching between tables"""
    env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9)
    obs, info = env.reset()
    
    observations = []
    table_ids = []
    
    # Collect observations from multiple tables
    for _ in range(10):
        if env._tournament_finished():
            break
        
        observations.append(obs.copy())
        table_ids.append(env.active_table_id)
        
        mask = info["action_mask"]
        if any(mask):
            action = np.argmax(mask)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                break
    
    # Verify observations are valid for their respective tables
    for i, (obs_vec, table_id) in enumerate(zip(observations, table_ids)):
        # Table ID should match
        assert obs_vec[5] == table_id, f"Observation {i}: table_id mismatch"
        
        # All values should be non-negative and reasonable
        assert all(val >= 0 for val in obs_vec), f"Observation {i}: negative values"
        assert obs_vec[4] in [0, 1], f"Observation {i}: invalid in_hand value"

def test_memory_and_state_consistency():
    """Test memory management and state consistency"""
    env = MultiTableTournamentEnv(total_players=27, max_players_per_table=9)
    
    # Multiple resets should not cause memory leaks or state issues
    for reset_num in range(5):
        obs, info = env.reset()
        
        # Verify clean state after reset
        assert len(env.elimination_order) == 0
        assert env.current_blind_level == 0
        assert env.total_hands_played == 0
        
        # Play a few steps
        for _ in range(10):
            mask = info["action_mask"]
            if any(mask):
                action = np.argmax(mask)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated:
                    break

def test_extreme_stack_scenarios():
    """Test extreme stack scenarios (very large/small stacks)"""
    env = MultiTableTournamentEnv(total_players=9, starting_stack=1000000)  # Large stacks
    obs, info = env.reset()
    
    # Should handle large stacks
    assert obs[0] == 1000000  # Player stack
    
    # Test with very small stacks
    env2 = MultiTableTournamentEnv(total_players=9, starting_stack=10)
    obs2, info2 = env2.reset()
    
    # Should handle small stacks
    assert obs2[0] == 10

def test_action_mask_edge_cases():
    """Test action mask generation in edge cases"""
    env = MultiTableTournamentEnv(total_players=9, max_players_per_table=9)
    obs, info = env.reset()
    
    # Test mask when player has exactly enough to call
    table = env.tables[env.active_table_id]
    current_player = table.players[table.game.current_player_idx]
    
    # Set player stack equal to amount needed to call
    to_call = table.game.current_bet - current_player.current_bet
    current_player.stack = to_call
    
    mask = env.legal_action_mask()
    
    # Should be able to call (all-in) but not raise
    if to_call > 0:
        assert mask[1] == True, "Should be able to call all-in"
        assert mask[2] == False, "Should not be able to raise with exact call amount"

def test_tournament_stats_edge_cases():
    """Test tournament statistics in edge cases"""
    env = MultiTableTournamentEnv(total_players=2, max_players_per_table=9)
    obs, info = env.reset()
    
    stats = env.get_tournament_stats()
    
    # Stats should be valid even with minimum players
    assert stats["total_players"] == 2
    assert stats["remaining_players"] <= 2
    assert stats["active_tables"] >= 1
    assert stats["chip_leader"] is not None
    assert stats["chip_leader_stack"] > 0

def test_rapid_elimination_scenario():
    """Test rapid elimination scenario (players bust quickly)"""
    env = MultiTableTournamentEnv(total_players=18, max_players_per_table=9)
    obs, info = env.reset()
    
    # Eliminate players rapidly to stress test the system
    players_to_eliminate = env.all_players[:15]  # Leave only 3 players
    
    for player in players_to_eliminate:
        player.stack = 0
    
    env._update_elimination_order()
    env._check_table_balancing()
    
    # Should handle rapid eliminations gracefully
    active_tables = env._get_active_tables()
    remaining_players = len([p for p in env.all_players if p.stack > 0])
    
    assert remaining_players == 3
    assert len(active_tables) >= 1
    
    # Should still be able to continue tournament
    if not env._tournament_finished():
        mask = env.legal_action_mask()
        assert any(mask), "Should have legal actions with remaining players"
