

# Poker AI Agent Development Plan


PRECONDITION:

1. Expand Test Coverage for Current Environment
Add more edge case tests to ensure flawless single-table logic:

Test all-in and split pot scenarios.
Test player elimination and placement rewards for every possible order.
Test blind increases and stack updates for long tournaments.
Test action mask correctness for every possible player state.
Test environment reset after every possible terminal state.
Test for correct dealer/SB/BB rotation after each hand.
2. Design Multi-Table Tournament Environment
Key requirements: 
(+ also change the environment to simulate a normal poker tournament with th blind structure and blinds increasing every NO number of hands)
Support N tables (e.g., 11 tables for 99 players, max 9 per table).
Track which players are at which table.
When a table drops below a threshold (e.g., 7 players), move players from other tables to balance.
Merge tables as the tournament progresses (e.g., down to 1 final table).
Ensure fair seat assignment and blind rotation after table balancing.
Track player elimination and placement across all tables.
3. Implement Multi-Table Tournament Logic
Create a new MultiTableTournamentEnv class.
Each table is a PokerTournamentEnv instance.
Central controller manages player distribution, table balancing, and merging.
Implement logic for moving players between tables and updating seat/blind positions.
Implement logic for merging tables as players are eliminated.
4. Write Comprehensive Multi-Table Tests
Test initial player distribution across tables.
Test table balancing after eliminations.
Test merging tables and final table formation.
Test correct blind and dealer assignment after player moves.
Test tournament end condition (one winner).
Test placement rewards for all players.
5. Teach Sharky to Play Multi-Table Tournaments
Update agent training scripts to support multi-table environments.
Add tests for agent behavior in multi-table scenarios.
6. Recommended Immediate Actions
Expand single-table tests as described above.
Design the API and data structures for multi-table tournament logic.
Implement a basic multi-table environment and add balancing/merging logic.
Write tests for every new feature.


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