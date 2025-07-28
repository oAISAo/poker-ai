#!/usr/bin/env python3

import time
from env.multi_table_tournament_env import MultiTableTournamentEnv

def main():
    env = MultiTableTournamentEnv(total_players=6, hands_per_blind_level=9)

    class SimpleTestAgent:
        def act(self, obs, action_mask=None, deterministic=True):
            if action_mask is None:
                return 1  # Default to call/check
            if action_mask[1]:  # Can call/check
                return 1
            elif action_mask[0]:  # Can fold
                return 0
            else:
                return 2  # Raise if only option

    test_agent = SimpleTestAgent()
    seed = int(time.time()) % (2**32 - 1)
    obs, info = env.reset(seed=seed)

    print("Running quick test to check for ActionValidationError...")
    
    # Run for just 100 steps to see if we get errors quickly
    for step in range(100):
        action_mask = info.get('action_mask', None)
        action = test_agent.act(obs, action_mask=action_mask)
        obs, reward, done, truncated, info = env.step(action)
        
        if step % 20 == 0:
            print(f"Step {step}: No errors so far")
            
        if done:
            print(f'Tournament finished successfully at step {step}')
            return True

    print('Test completed - no ActionValidationErrors detected in 100 steps')
    return False

if __name__ == "__main__":
    main()
