


Start AI environment:
find . -name "__pycache__" -exec rm -rf {} +
deactivate
source ../pokerai/bin/activate


tree -L 2 -I '__pycache__|.git|.pytest_cache|*.egg-info|*.pyc|.DS_Store|venv|env' --dirsfirst

