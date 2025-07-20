from agents.sharky_agent import SharkyAgent

def test_sharky_action_selection():
    sharky = SharkyAgent()
    state = {"legal_actions": ["fold", "call", "raise"]}
    action = sharky.get_action(state)
    assert action in state["legal_actions"]

def test_sharky_training_log(caplog):
    sharky = SharkyAgent()
    with caplog.at_level("INFO"):
        sharky.train(env=None, timesteps=5)
    assert "Starting dummy training" in caplog.text
    assert "Dummy training complete" in caplog.text
    assert sharky.is_trained
