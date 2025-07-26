# 🧠 Poker AI Project Plan  
_Last updated: 2025-07-26_  
_Author: Aiša Berro_

---

## 🎯 Project Goal  
Build a modular, extensible AI system that learns to play No-Limit Texas Hold’em poker through self-play, reinforcement learning, and imitation learning. The system will support multiple agent personalities, robust evaluation, and easy extension for future research and experimentation.

---

## 👩‍💻 User Profile  
- **Name:** Aiša Berro  
- **Background:** Computer Science, Frontend Developer (JS, Python)  
- **Interest:** AI, RL, Poker  
- **Motivation:** Personal research, not commercial  
- **Budget:** Flexible, focused on quality and learning

---

## 🧠 AI Agents  
Develop and evolve multiple agents, each with a unique strategy:

- **Nashy:** Nash-equilibrium conservative agent  
- **Bluffy:** Aggressive, bluffing-focused agent  
- **Sharky:** RL-based self-improver (PPO/A2C)  
- **Basey:** Rule-based benchmark agent  
- **Simon:** Human-influenced agent trained from real hand history

Agents will compete in various environments (cash, SNG, MTT) and a multi-agent tournament system.

---

## 📁 Folder Structure
```bash
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

## 🛠️ Tech Stack
- Python 3.11+
- Stable Baselines3, sb3-contrib
- OpenAI Gym/Gymnasium
- PyPokerEngine / custom engine
- Pytest
- VS Code
- macOS (Intel/Apple Silicon)
- (Optional: CUDA/GPU/cloud training)

---

## ✅ Current Progress
- Virtualenv and dependencies set up
- Folder structure scaffolded
- Poker engine implemented (cards, betting, hand evaluation)
- Base agents and rule-based logic tested
- RL agent (Sharky) training and saving implemented
- Evaluation and metrics logging in place

---

## 📌 Next Steps
1. **Agent Evolution:**  
   - Train and version Sharky (`sharky_1.0.0`, `sharky_1.0.1`, ...), run tournaments, and evolve best agents.
2. **Human Imitation Agent:**  
   - Parse Simon’s hand history, train `Simon_basic` via supervised learning, then evolve with RL.
3. **Mixed Tournaments:**  
   - Run tournaments with all agent versions, log and analyze results.
4. **Documentation & Usability:**  
   - Add docstrings and usage examples to all public classes/functions.
   - Update `README.md` with setup, training, evaluation, and extension instructions.
5. **Testing & Validation:**  
   - Expand unit tests and integration tests for all modules.
6. **Visualization:**  
   - Plot training/evaluation metrics for agent progress tracking.

---

## 🚀 Best Practices
- Save models and logs with clear versioning.
- Visualize training and tournament results.
- Periodically retrain and evaluate agents against each other.
- Use both RL and supervised learning for robust agent development.
- Maintain up-to-date documentation and onboarding guides.

---

## 📓 Notes
- This file is for project context and continuity.
- Update regularly as the project evolves.
- Use as a reference for onboarding, planning, and retrospectives.