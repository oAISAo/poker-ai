

# Poker AI Agent Development Plan



## Phase 1: Sharky Self-Play Evolution
1. Train Sharky from scratch, save as `sharky_1.0.0`.
2. Continue training, save as `sharky_1.0.1`, ..., up to `sharky_1.0.9`.
3. Run a tournament with all Sharky versions (`sharky_1.0.0` to `sharky_1.0.9`).
4. Analyze results, select the best-performing agent or ensemble, and save as `sharky_1.1.0`.
5. Repeat the process for further evolution (`sharky_1.1.x`, etc.).

## Phase 2: Simon Agent from Hand History
1. Parse Simon’s hand history and train a supervised agent (`Simon_basic`) to mimic Simon’s play.
2. Use `Simon_basic` as the starting point for RL training, save as `Simon_1.0.0`.
3. Train `Simon_1.0.0` via RL, playing against Sharky versions.
4. Save new versions as `Simon_1.0.1`, ..., and run tournaments for evaluation.

## Phase 3: Mixed Tournaments and Analysis
1. Run tournaments with Sharky and Simon versions.
2. Log and analyze placements, rewards, and strategies.
3. Use results to guide further training, tuning, and agent development.

## Best Practices
- Save all models and logs with clear versioning.
- Visualize training and tournament results.
- Periodically retrain and evaluate agents against each other.
- Use both RL and supervised learning for robust agent development.