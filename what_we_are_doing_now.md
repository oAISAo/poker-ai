🟡 What’s In Progress / Next Steps

1. Tournament Structure
Start with 9 players, each with a starting stack (e.g., 1000 chips).
Play hands in sequence until only one player remains (all others are eliminated).
Track player elimination order for placement-based rewards.
2. Blinds Schedule
Define a blinds schedule (e.g., every X hands, increase blinds).
Update the environment/game to increase blinds at the right intervals.
3. Environment Changes
Tournament loop:
After each hand, check for eliminated players (stack == 0).
Remove eliminated players from the game.
Continue until one player remains.
Blinds logic:
At the start of each hand, check if it’s time to increase blinds.
Update small_blind and big_blind accordingly.
Placement tracking:
Record the order in which players are eliminated.
Assign rewards at the end based on placement (e.g., 1st place = highest reward, 2nd = less, etc.).


Add CLI/config options for agent selection, training length, and evaluation.
Log results to logger.py.

4. Evaluation and Metrics
Goal: Track agent performance over time and across matchups.
Action:
Implement win-rate, profit/loss, and showdown stats.
Add tournament simulation scripts for multi-agent evaluation.

5. Continuous Testing
Goal: Ensure all new features are robust and bug-free.
Action:
Expand test to cover new RL agents and environment logic.
Add regression tests for any bugs found during RL training.

6. Documentation and Usability
Goal: Make the project easy to use and extend.
Action:
Add docstrings and usage examples to all public classes/functions.
Update README.md with setup, training, and evaluation instructions.