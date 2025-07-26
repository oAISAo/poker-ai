# from agents.sharky_agent import SharkyAgent
# from env.poker_env import PokerEnv

# def test_sharky_action_selection():
#     env = PokerEnv(num_players=2)
#     sharky = SharkyAgent(env)
#     action = sharky.act(env.reset())
#     assert isinstance(action, (int, np.integer)), f"Action should be int, got {type(action)}"
#     assert 0 <= action < env.action_space.n, f"Action {action} out of bounds"

# def test_sharky_training_log(caplog):
#     env = PokerEnv(num_players=2)
#     sharky = SharkyAgent(env)
#     with caplog.at_level("INFO"):
#         # If SharkyAgent has a dummy train method, call it; otherwise, call learn
#         if hasattr(sharky, "train"):
#             sharky.train(env, timesteps=5)
#         else:
#             sharky.learn(total_timesteps=5)
#     # These asserts may need to be updated if the logging output has changed
#     assert "Starting dummy training" in caplog.text or "Training complete" in caplog.text
#     assert "Dummy training complete" in caplog.text or "Training complete" in caplog.text
#     assert getattr(sharky, "is_trained", True) or hasattr(sharky, "model")