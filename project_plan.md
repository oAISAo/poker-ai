🧠 Poker AI Project Plan  
Last updated: 2025-07-20  
Author: Aiša Berro (for ChatGPT continuity)

---

🎯 Project Goal  
Create a high-performing AI system that learns how to play poker (starting with No-Limit Texas Hold’em) through self-play and reinforcement learning. The system will simulate different AI personalities (agents) and environments to train the ultimate poker decision-maker that can advise the user in real tournaments.

---

👩‍💻 User Profile  
Name: Aiša Berro  
Background: Computer Science, Frontend Developer (strong JS, some Python)  
Interest: AI, Reinforcement Learning, Poker  
Motivation: Personal project, not commercial.  
Budget: No limit (but cautious about quality and timing of purchases)

---

🧠 AI Agents  
We are developing 4 separate AI agents, each with its own learning strategy and personality. Each will train and evolve over time:

- **Nashy** – Nash-equilibrium-based conservative agent  
- **Bluffy** – Aggressive, bluffing-heavy agent  
- **Sharky** – Optimal, learning-based self-improver (trained with RL)  
- **Basey** – Rule-based agent with fixed, non-learning strategy (for benchmarking)  
🆕 - **Simon** – (planned) Human-influenced agent that learns from real player Simon via logs or interactions.

Agents will be trained in different environments and game types to evaluate their performance across various contexts (cash games, SNGs, MTTs, etc.).

🆕 A simulated **multi-agent tournament system** is also planned to compare and evolve agents in a competitive setting.

---

📁 Folder Structure
```bash
poker-ai/
├── agents/          # Each AI agent and base class
│   ├── base_agent.py
│   ├── nash.py
│   ├── bluffy.py
│   ├── sharky.py
│   └── basey.py      # 🆕 Rule-based test agent
├── engine/          # Core poker logic
│   ├── game.py
│   ├── player.py
│   ├── cards.py
│   └── evaluator.py
├── env/             # Gym-compatible training environment
│   └── poker_env.py
├── train/           # Training scripts
│   └── train_agents.py
├── utils/           # Logging, helpers, wrappers
│   └── logger.py
├── main.py          # Script to run games or simulations
├── requirements.txt
└── pytest.ini        # 🆕 Testing config for pytest
🛠️ Tech Stack

Python 3.11+

Stable Baselines3

OpenAI Gym

PyPokerEngine / custom engine

Pytest 🆕

VS Code (optional)

macOS (initially on Intel MacBook Pro, upgrades later)

(Possibly later: CUDA/GPU training on cloud or new Mac)

✅ Current Progress

✅ Virtualenv created (pokerai)
✅ Python installed
✅ Stable Baselines3 installed
✅ Folder structure scaffolded
✅ Game engine implemented (cards, game flow, betting)
✅ hand_evaluator.py working with correct ranking logic
✅ Showdown display improved with descriptive ranks
✅ Best 5-card hand shown per player
✅ basey.py implemented and tested 🆕
✅ test/test_agents.py created with unit test for Basey 🆕

❌ RL agents coded
❌ Training scripts created
❌ Evaluation/metrics
❌ Final advisor model

📌 Next Steps

🆕 Design general-purpose RL agent class as foundation for agents like Sharky
🆕 Choose episode-based learning loop to start; consider step-based later
🆕 Modularize reward, observation, and action-space design
🆕 Implement Gym environment in poker_env.py
🆕 Refactor main.py to support modular agents and evaluation
🆕 Use pytest for continuous bug-checking and testing

Define BaseAgent interface in base_agent.py ✅

Create Basey rule-based agent for benchmarking ✅

Begin training Sharky (PPO/A2C) via self-play

Evaluate each agent against others in head-to-head play

Continuously refine logic and retrain

Build long-term tournament arena and meta-learning simulation

📓 Notes

This file is for ChatGPT context restoration in case of lost session/memory.
Paste it at the start of a new session to reestablish context.
Update it as the project evolves.