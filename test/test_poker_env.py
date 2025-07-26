import pytest
from env.poker_env import PokerEnv

# def legal_actions(env):
#     player = env.players[env.game.current_player_idx]
#     to_call = env.game.current_bet - player.current_bet
#     actions = []
#     if to_call == 0:
#         actions = [1, 2]  # check or raise
#     else:
#         actions = [0, 1, 2]  # fold, call, raise
#     return actions

# def test_env_reset_and_step():
#     env = PokerEnv(num_players=2)
#     obs = env.reset()
#     assert obs.shape == (5,)
#     done = False
#     for _ in range(10):
#         actions = legal_actions(env)
#         action = actions[0]  # Always pick the first legal action for determinism, or use random.choice(actions)
#         obs, reward, done, info = env.step(action)
#         assert obs.shape == (5,)
#         if done:
#             break
#     assert isinstance(done, bool)