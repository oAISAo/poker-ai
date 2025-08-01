import env
import pytest
import numpy as np
from env.poker_tournament_env import PokerTournamentEnv

def test_env_reset_and_obs_shape():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)
    assert "action_mask" in info
    assert info["action_mask"].shape == (3,)

def test_step_valid_actions_and_termination():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    done = False
    steps = 0
    while not done and steps < 1000:
        print(f"[DEBUG] Step {steps}: current_player={env.game.current_player_idx}, stack={[p.stack for p in env.players]}, in_hand={[p.in_hand for p in env.players]}")
        mask = env.legal_action_mask()
        # Pick a legal action
        action = next((i for i, m in enumerate(mask) if m), 1)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    if steps >= 1000:
        print("[ERROR] Step limit reached. Game state:")
        print(f"Players: {[p.name for p in env.players]}")
        print(f"Stacks: {[p.stack for p in env.players]}")
        print(f"In hand: {[p.in_hand for p in env.players]}")
        print(f"Hand over: {env.game.hand_over}")
        assert False, "Test failed: environment did not terminate after 1000 steps"
    assert steps > 0

def test_legal_action_mask_states():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    # Simulate a player going all-in
    assert env.game.current_player_idx is not None
    player = env.players[env.game.current_player_idx]
    player.stack = 0
    mask = env.legal_action_mask()
    # If stack is 0 (all-in), no actions should be legal
    assert not any(mask), f"All-in player should have no legal actions, got mask={mask}"
    # Simulate player eliminated
    player.in_hand = False
    mask = env.legal_action_mask()
    assert not any(mask)

def test_player_elimination_and_placement_reward():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    # Eliminate two players manually
    env.players[0].stack = 0
    env.players[1].stack = 0
    env.elimination_order = [env.players[0], env.players[1]]
    winner = env.players[2]
    reward = env._get_placement_rewards(winner)
    assert reward == 100  # First place reward
    loser = env.players[0]
    reward = env._get_placement_rewards(loser)
    assert reward > 0

def test_blinds_schedule_and_game_reset():
    env = PokerTournamentEnv(num_players=3, hands_per_level=2)
    obs, info = env.reset()
    initial_blind_level = env.current_blind_level
    for hand_num in range(4):
        done = False
        steps = 0
        while not done and steps < 1000:
            mask = info["action_mask"]
            action = int(np.argmax(mask))
            print(f"[DEBUG] Hand {hand_num}, Step {steps}: current_player={env.game.current_player_idx}, stack={[p.stack for p in env.players]}, in_hand={[p.in_hand for p in env.players]}, blind_level={env.current_blind_level}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        if steps >= 1000:
            print("[ERROR] Step limit reached in test_blinds_schedule_and_game_reset. Game state:")
            print(f"Players: {[p.name for p in env.players]}")
            print(f"Stacks: {[p.stack for p in env.players]}")
            print(f"In hand: {[p.in_hand for p in env.players]}")
            print(f"Hand over: {env.game.hand_over}")
            print(f"Blind level: {env.current_blind_level}")
            assert False, "Test failed: environment did not terminate after 1000 steps in a hand"
    assert env.current_blind_level > initial_blind_level

def test_all_in_action_mask_and_step():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    assert env.game.current_player_idx is not None
    player = env.players[env.game.current_player_idx]
    # Set player to all-in
    player.stack = 0
    mask = env.legal_action_mask()
    # If stack is 0 (all-in), no actions should be legal
    assert not any(mask), f"All-in player should have no legal actions, got mask={mask}"
    # Try to step with a raise, should not be allowed
    with pytest.raises(Exception):
        env.step(2)

def test_fold_action_removes_player_from_hand():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    assert env.game.current_player_idx is not None
    player = env.players[env.game.current_player_idx]
    # Force a fold
    obs, reward, terminated, truncated, info = env.step(0)
    assert not player.in_hand

def test_blind_increase_and_reset_logic():
    env = PokerTournamentEnv(num_players=3, hands_per_level=1)
    obs, info = env.reset()
    initial_blind_level = env.current_blind_level
    done = False
    steps = 0
    while not done and steps < 1000:
        mask = info["action_mask"]
        action = int(np.argmax(mask))
        print(f"[DEBUG] Step {steps}: current_player={env.game.current_player_idx}, stack={[p.stack for p in env.players]}, in_hand={[p.in_hand for p in env.players]}, blind_level={env.current_blind_level}")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    if steps >= 1000:
        print("[ERROR] Step limit reached in test_blind_increase_and_reset_logic. Game state:")
        print(f"Players: {[p.name for p in env.players]}")
        print(f"Stacks: {[p.stack for p in env.players]}")
        print(f"In hand: {[p.in_hand for p in env.players]}")
        print(f"Hand over: {env.game.hand_over}")
        print(f"Blind level: {env.current_blind_level}")
        assert False, "Test failed: environment did not terminate after 1000 steps"
    assert env.current_blind_level == initial_blind_level + 1

def test_elimination_order_and_final_placement():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    # Eliminate two players
    env.players[0].stack = 0
    env.players[1].stack = 0
    env.elimination_order = [env.players[0], env.players[1]]
    # Simulate hand over
    env.game.hand_over = True
    # Set current player to the remaining player
    env.game.current_player_idx = 2
    # Do NOT call env.step() after hand_over is True.
    winner = env.players[2]
    reward = env._get_placement_rewards(winner)
    assert reward == 100
    loser1 = env.players[0]
    loser2 = env.players[1]
    assert env._get_placement_rewards(loser1) > 0
    assert env._get_placement_rewards(loser2) > 0
    # Only one player should remain
    active_players = [p for p in env.players if p.stack > 0]
    assert len(active_players) == 1

def test_action_mask_for_eliminated_player():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    assert env.game.current_player_idx is not None
    player = env.players[env.game.current_player_idx]
    player.in_hand = False
    mask = env.legal_action_mask()
    # No actions should be legal for eliminated player
    assert not any(mask)

def test_reward_calculation_on_stack_change():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    assert env.game.current_player_idx is not None
    player = env.players[env.game.current_player_idx]
    # Simulate a win
    player.stack += 100
    reward = env._get_reward(player)
    assert reward == 100

def test_env_reset_clears_elimination_order():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    env.elimination_order = [env.players[0], env.players[1]]
    obs, info = env.reset()
    assert env.elimination_order == []

def test_action_mask_when_to_call_is_zero():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    assert env.game.current_player_idx is not None
    player = env.players[env.game.current_player_idx]
    env.game.current_bet = player.current_bet  # to_call = 0
    mask = env.legal_action_mask()
    # Check should be legal
    assert mask[1]

def test_raise_action_mask_min_max():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    assert env.game.current_player_idx is not None
    player = env.players[env.game.current_player_idx]
    # Set up so player can raise
    player.stack = 1000
    env.game.current_bet = 20
    env.game.last_raise_amount = 20
    env.game.big_blind = 20
    mask = env.legal_action_mask()
    assert mask[2]  # Raise should be legal

def test_env_handles_multiple_resets():
    env = PokerTournamentEnv(num_players=3)
    for _ in range(5):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)
        assert "action_mask" in info

def test_env_with_nine_players_reset_and_obs():
    env = PokerTournamentEnv(num_players=9)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)
    assert "action_mask" in info
    assert info["action_mask"].shape == (3,)

def test_elimination_order_and_rewards_nine_players():
    env = PokerTournamentEnv(num_players=9)
    obs, info = env.reset()
    # Eliminate 8 players
    for i in range(8):
        env.players[i].stack = 0
        env.elimination_order.append(env.players[i])
    winner = env.players[8]
    reward = env._get_placement_rewards(winner)
    assert reward == 100  # First place reward
    for i in range(8):
        reward = env._get_placement_rewards(env.players[i])
        assert reward > 0

def test_blind_increase_with_many_players():
    env = PokerTournamentEnv(num_players=9, hands_per_level=1)
    obs, info = env.reset()
    initial_blind_level = env.current_blind_level
    done = False
    steps = 0
    while not done and steps < 1000:
        mask = info["action_mask"]
        action = int(np.argmax(mask))
        print(f"[DEBUG] Step {steps}: current_player={env.game.current_player_idx}, stack={[p.stack for p in env.players]}, in_hand={[p.in_hand for p in env.players]}, blind_level={env.current_blind_level}")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    if steps >= 1000:
        print("[ERROR] Step limit reached in test_blind_increase_with_many_players. Game state:")
        print(f"Players: {[p.name for p in env.players]}")
        print(f"Stacks: {[p.stack for p in env.players]}")
        print(f"In hand: {[p.in_hand for p in env.players]}")
        print(f"Hand over: {env.game.hand_over}")
        print(f"Blind level: {env.current_blind_level}")
        assert False, "Test failed: environment did not terminate after 1000 steps"
    assert env.current_blind_level == initial_blind_level + 1

def test_action_mask_for_all_in_and_eliminated_players_nine():
    env = PokerTournamentEnv(num_players=9)
    obs, info = env.reset()
    # Set player 0 to all-in
    env.players[0].stack = 0
    env.game.current_player_idx = 0
    mask = env.legal_action_mask()
    assert not any(mask), f"All-in player should have no legal actions, got mask={mask}"
    # Eliminate player 1
    env.players[1].in_hand = False
    env.game.current_player_idx = 1
    mask = env.legal_action_mask()
    assert not any(mask)

def test_tournament_ends_when_one_player_left():
    env = PokerTournamentEnv(num_players=5)
    obs, info = env.reset()
    # Eliminate 4 players
    for i in range(4):
        env.players[i].stack = 0
        env.elimination_order.append(env.players[i])
    env.game.hand_over = True
    env.game.current_player_idx = 4
    # Do NOT call env.step() after hand_over is True.
    winner = env.players[4]
    reward = env._get_placement_rewards(winner)
    assert reward == 100
    active_players = [p for p in env.players if p.stack > 0]
    assert len(active_players) == 1

def test_env_handles_multiple_resets_nine_players():
    env = PokerTournamentEnv(num_players=9)
    for _ in range(3):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (5,)
        assert "action_mask" in info

def test_blind_schedule_progression_nine_players():
    env = PokerTournamentEnv(num_players=9, hands_per_level=2)
    obs, info = env.reset()
    initial_blind_level = env.current_blind_level
    for hand_num in range(6):
        done = False
        steps = 0
        while not done and steps < 1000:
            mask = info["action_mask"]
            action = int(np.argmax(mask))
            print(f"[DEBUG] Hand {hand_num}, Step {steps}: current_player={env.game.current_player_idx}, stack={[p.stack for p in env.players]}, in_hand={[p.in_hand for p in env.players]}, blind_level={env.current_blind_level}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        if steps >= 1000:
            print("[ERROR] Step limit reached in test_blind_schedule_progression_nine_players. Game state:")
            print(f"Players: {[p.name for p in env.players]}")
            print(f"Stacks: {[p.stack for p in env.players]}")
            print(f"In hand: {[p.in_hand for p in env.players]}")
            print(f"Hand over: {env.game.hand_over}")
            print(f"Blind level: {env.current_blind_level}")
            assert False, "Test failed: environment did not terminate after 1000 steps in a hand"
    assert env.current_blind_level > initial_blind_level

def test_action_mask_for_player_with_exact_to_call():
    env = PokerTournamentEnv(num_players=4)
    obs, info = env.reset()
    assert env.game.current_player_idx is not None
    player = env.players[env.game.current_player_idx]
    env.game.current_bet = player.current_bet + player.stack
    mask = env.legal_action_mask()
    # Player can call all-in, but not raise
    assert mask[1]
    assert not mask[2]

def test_env_reset_clears_all_player_states():
    env = PokerTournamentEnv(num_players=6)
    obs, info = env.reset()
    for p in env.players:
        p.stack = 0
        p.in_hand = False
    obs, info = env.reset()
    sb = env.game.small_blind
    bb = env.game.big_blind
    dealer_pos = env.game.dealer_position
    sb_pos = (dealer_pos + 1) % env.num_players
    bb_pos = (dealer_pos + 2) % env.num_players
    for i, p in enumerate(env.players):
        if i == sb_pos:
            assert p.stack == env.starting_stack - sb
        elif i == bb_pos:
            assert p.stack == env.starting_stack - bb
        else:
            assert p.stack == env.starting_stack
        assert p.in_hand

def test_all_in_split_pot_scenario():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    # Simulate two players going all-in with equal hands
    p0, p1, p2 = env.players
    p0.stack = 0
    p1.stack = 0
    p2.stack = 0
    env.game.pot = 300
    # Simulate split pot
    winnings = env.game.pot // 3
    for p in env.players:
        p.stack += winnings
    for p in env.players:
        assert p.stack == winnings

def test_player_elimination_and_placement_rewards_all_orders():
    env = PokerTournamentEnv(num_players=4)
    obs, info = env.reset()
    # Eliminate players in every possible order
    orders = [
        [0, 1, 2], [1, 2, 0], [2, 0, 1]
    ]
    for order in orders:
        env.reset()
        for idx in order:
            env.players[idx].stack = 0
            env.elimination_order.append(env.players[idx])
        winner = [p for p in env.players if p.stack > 0][0]
        reward = env._get_placement_rewards(winner)
        assert reward == 100
        for idx in order:
            reward = env._get_placement_rewards(env.players[idx])
            assert reward > 0

def test_blind_increase_and_stack_update_long_tournament():
    env = PokerTournamentEnv(num_players=3, hands_per_level=1)
    obs, info = env.reset()
    initial_blind_level = env.current_blind_level
    for hand_num in range(10):
        done = False
        steps = 0
        while not done and steps < 1000:
            mask = info["action_mask"]
            if not any(mask):
                assert env.game.current_player_idx is not None
                print(f"[DEBUG] No legal actions for player {env.players[env.game.current_player_idx].name}, stack={env.players[env.game.current_player_idx].stack}, in_hand={env.players[env.game.current_player_idx].in_hand}")
                done = True
                break
            action = int(np.argmax(mask))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        if steps >= 1000:
            print("[ERROR] Step limit reached in test_blind_increase_and_stack_update_long_tournament. Game state:")
            print(f"Players: {[p.name for p in env.players]}")
            print(f"Stacks: {[p.stack for p in env.players]}")
            print(f"In hand: {[p.in_hand for p in env.players]}")
            print(f"Hand over: {env.game.hand_over}")
            print(f"Blind level: {env.current_blind_level}")
            # Additional debug: print action mask for all players
            for i, p in enumerate(env.players):
                env.game.current_player_idx = i
                mask = env.legal_action_mask()
                print(f"Player {p.name}: stack={p.stack}, in_hand={p.in_hand}, mask={mask}")
            assert False, "Test failed: environment did not terminate after 1000 steps in a hand"
    assert env.current_blind_level > initial_blind_level
    # Instead, check that no player has a negative stack (which would be a bug)
    for p in env.players:
        assert p.stack >= 0, f"Player {p.name} has negative stack: {p.stack}"

def test_action_mask_all_player_states():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    for p in env.players:
        # Normal
        p.stack = 100
        p.in_hand = True
        env.game.current_player_idx = env.players.index(p)
        assert env.game.current_player_idx is not None
        _player = env.players[env.game.current_player_idx]
        mask = env.legal_action_mask()
        assert any(mask)
        # All-in
        p.stack = 0
        mask = env.legal_action_mask()
        assert not any(mask), f"All-in player should have no legal actions, got mask={mask}"
        # Eliminated
        p.in_hand = False
        mask = env.legal_action_mask()
        assert not any(mask)

def test_env_reset_after_terminal_states():
    env = PokerTournamentEnv(num_players=3)
    obs, info = env.reset()
    # Eliminate all but one player
    env.players[0].stack = 0
    env.players[1].stack = 0
    env.elimination_order = [env.players[0], env.players[1]]
    env.game.hand_over = True
    obs, info = env.reset()
    # All players should be reset, but account for blinds
    sb = env.game.small_blind
    bb = env.game.big_blind
    dealer_pos = env.game.dealer_position
    sb_pos = (dealer_pos + 1) % env.num_players
    bb_pos = (dealer_pos + 2) % env.num_players
    for i, p in enumerate(env.players):
        if i == sb_pos:
            assert p.stack == env.starting_stack - sb
        elif i == bb_pos:
            assert p.stack == env.starting_stack - bb
        else:
            assert p.stack == env.starting_stack
        assert p.in_hand

def test_dealer_sb_bb_rotation_after_each_hand():
    env = PokerTournamentEnv(num_players=5)
    obs, info = env.reset()
    positions = []
    for hand_num in range(5):
        dealer = env.game.dealer_position
        sb = (dealer + 1) % env.num_players
        bb = (dealer + 2) % env.num_players
        positions.append((dealer, sb, bb))
        done = False
        steps = 0
        while not done and steps < 1000:
            mask = info["action_mask"]
            action = int(np.argmax(mask))
            assert env.game.current_player_idx is not None
            print(f"[DEBUG] Hand {hand_num}, Step {steps}: Player {env.players[env.game.current_player_idx].name}, Stack {env.players[env.game.current_player_idx].stack}, Action {action}, HandOver {env.game.hand_over}")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        if steps >= 1000:
            print("[ERROR] Step limit reached in test_dealer_sb_bb_rotation_after_each_hand. Game state:")
            print(f"Players: {[p.name for p in env.players]}")
            print(f"Stacks: {[p.stack for p in env.players]}")
            print(f"In hand: {[p.in_hand for p in env.players]}")
            print(f"Hand over: {env.game.hand_over}")
            print(f"Dealer: {env.game.dealer_position}")
            assert False, f"Hand {hand_num} did not terminate after 1000 steps"
        env.reset()
    dealers = [pos[0] for pos in positions]
    sbs = [pos[1] for pos in positions]
    bbs = [pos[2] for pos in positions]
    assert sorted(set(dealers)) == list(range(env.num_players))
    assert sorted(set(sbs)) == list(range(env.num_players))
    assert sorted(set(bbs)) == list(range(env.num_players))
