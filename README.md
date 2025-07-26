# 🧠 Poker AI

A modular, extensible AI system for learning No-Limit Texas Hold’em poker through self-play, reinforcement learning, and imitation learning. Supports multiple agent personalities, robust evaluation, and easy extension for research and experimentation.

---

## 🚀 Features

- Modular poker engine with full Texas Hold’em rules
- Multiple agent personalities (RL, rule-based, human-mimic)
- Gym-compatible environments for RL training
- Tournament system for agent evaluation
- Model saving/loading and versioning
- Easy-to-extend architecture

---

## 📦 Folder Structure

```
poker-ai/
├── agents/          # Agent classes and personalities
├── engine/          # Poker game logic
├── env/             # Gym-compatible environments
├── train/           # Training scripts
├── utils/           # Logging, helpers
├── main.py          # Entry point for games/simulations
├── requirements.txt
├── pytest.ini       # Testing config
└── README.md        # Project documentation
```

---

## 🛠️ Setup

1. **Clone the repo:**
   ```sh
   git clone https://github.com/oAISAo/poker-ai.git
   cd poker-ai
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python3 -m venv pokerai
   source pokerai/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## 🏃‍♂️ Training Agents

### Train Sharky from scratch and save:
```sh
python -m train.train_agents --agent sharky --timesteps 20000 --eval-episodes 20 --log INFO
```

### Continue training from a saved model:
```sh
python -m train.train_agents --agent sharky --timesteps 20000 --eval-episodes 20 --log INFO --load-model sharky_model.zip
```

### Model versioning:
- Save models as `sharky_1.0.0.zip`, `sharky_1.0.1.zip`, etc.
- Run tournaments between versions to select/evolve the best agent.

---

## 🧑‍💻 Usage Examples

### Run a tournament between agents:
```python
from env.poker_tournament_env import PokerTournamentEnv
from agents.sharky_agent import SharkyAgent

env = PokerTournamentEnv(num_players=9, starting_stack=1000)
agent = SharkyAgent(env, use_maskable_ppo=True)
agent.model = MaskablePPO.load("sharky_1.0.0.zip", env=env)
# Play or evaluate as needed
```

### Add a new agent:
- Create a new file in `agents/` (e.g., `simon_agent.py`)
- Implement the agent class, following the `BaseRLAgent` interface
- Add training and evaluation scripts as needed

---

## 📝 Documentation & Extensibility

- All public classes and functions include docstrings and usage examples.
- See `project_plan.md` for development roadmap and context.
- Extend agents, environments, or training scripts by following the modular structure.

---

## 🧪 Testing

- Run unit tests with pytest:
  ```sh
  pytest
  ```

---

## 📊 Visualization

- Training and evaluation metrics are logged.
- Use Matplotlib or TensorBoard to plot agent progress over time.

---

## 🤝 Contributing

- Fork the repo and submit pull requests.
- Please add docstrings and tests for new features.

---

## 📓 Notes

- This project is for research and learning.
- Update documentation and logs regularly.
- For questions or ideas, open an issue or contact Aiša Berro.

---

## 🏆 Project Goal

Create the ultimate poker AI advisor through continuous agent evolution, robust evaluation, and human-influenced learning.
