
Restart AI environment:
find . -name "__pycache__" -exec rm -rf {} +
deactivate
source ../pokerai/bin/activate


tree -L 2 -I '__pycache__|.git|.pytest_cache|*.egg-info|*.pyc|.DS_Store|venv|env' --dirsfirst


To train and save:
python -m train.train_agents --agent sharky --timesteps 20000 --eval-episodes 20 --log INFO
To continue training from a saved model:
python -m train.train_agents --agent sharky --timesteps 20000 --eval-episodes 20 --log INFO --load-model sharky_model.zip
(Use the first command the very first time you train Sharky, or if you want to start over.
Use the second command if you want to continue training Sharky from a previous session, building on what it has already learned.)

