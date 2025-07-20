🧠 Poker AI Project Plan
Last updated: 2025-07-19
Author: Aiša Berro (for ChatGPT continuity)

🎯 Project Goal
Create a high-performing AI system that learns how to play poker (starting with No-Limit Texas Hold’em) through self-play and reinforcement learning. The system will simulate different AI personalities (agents) and environments to train the ultimate poker decision-maker that can advise the user in real tournaments.

👩‍💻 User Profile
Name: Aiša Berro
Background: Computer Science, Frontend Developer (strong JS, some Python)
Interest: AI, Reinforcement Learning, Poker
Motivation: Personal project, not commercial.
Budget: No limit (but cautious about quality and timing of purchases)

🧠 AI Agents
We are developing 4 separate AI agents, each with its own learning strategy and personality. Each will train and evolve over time:

Nashy – Nash-equilibrium-based conservative agent

Bluffy – Aggressive, bluffing-heavy agent

Sharky – Optimal, learning-based self-improver (trained with RL)

Basey – Rule-based agent with fixed, non-learning strategy (for benchmarking)

Agents will be trained in different environments and game types to evaluate their performance across various contexts (cash games, SNGs, MTTs, etc.).

📁 Folder Structure

bash
Copy
Edit
poker-ai/
├── agents/          # Each AI agent and base class
│   ├── base_agent.py
│   ├── nash.py
│   ├── bluffy.py
│   └── sharky.py
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
└── requirements.txt
🛠️ Tech Stack

Python 3.11+

Stable Baselines3

OpenAI Gym

PyPokerEngine / custom engine

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

❌ Agents coded

❌ Training scripts created

❌ Evaluation/metrics

❌ Final advisor model

📌 Next Steps

Build poker_env.py for Gym compatibility

Implement base_agent.py as abstract class/interface

Define simple Basey agent as benchmark

Begin training Sharky with PPO/A2C in self-play mode

Evaluate each agent against others

Continuously refine logic and retrain

📝 Notes
This file is for ChatGPT context restoration in case of lost session/memory.
Paste it at the start of a new session to reestablish context.
Update it as the project evolves.