# test/test_poker_env.py

import pytest
from env.poker_env import PokerEnv

def test_reset_and_step():
    env = PokerEnv()
    obs = env.reset()
    assert obs.shape == (10,)
    done = False
    total_reward = 0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    assert isinstance(total_reward, (int, float))
    assert done is True or done is False
