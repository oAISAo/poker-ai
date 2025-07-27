
Restart AI environment:
find . -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
deactivate
source ../pokerai/bin/activate


tree -L 2 -I '__pycache__|.git|.pytest_cache|*.egg-info|*.pyc|.DS_Store|venv|env' --dirsfirst


To train and save:
python -m train.train_agents --agent sharky --timesteps 20000 --eval-episodes 20 --log INFO
To continue training from a saved model:
python -m train.train_agents --agent sharky --timesteps 20000 --eval-episodes 20 --log INFO --load-model sharky_model.zip
(Use the first command the very first time you train Sharky, or if you want to start over.
Use the second command if you want to continue training Sharky from a previous session, building on what it has already learned.)


CHECK TRAINING STATS:

1. make file executable
chmod +x sharky.sh

2. Nice table of all Sharky versions and whether they're trained/evaluated.
./sharky.sh status

3. Raw training data (timesteps completed, etc.)
./sharky.sh stats 1.0.0

4. Evaluate performance
./sharky.sh evaluate 1.0.0


What each result means:
🏆 Tournament Results:

Average Placement: In an 18-player tournament, 1st = best, 18th = worst

Good: 1-6 (top third)
Okay: 7-12 (middle third)
Needs work: 13-18 (bottom third)
Win Rate: Percentage of tournaments won

Excellent: 20%+ (way above random 5.6%)
Good: 10-20%
Learning: 5-10%
Struggling: 0-5%
Average Reward: Points earned (our reward system)

Winner gets 1000 points
2nd place gets 600 points
etc.