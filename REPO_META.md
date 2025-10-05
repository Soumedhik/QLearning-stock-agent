# Repository: rl-stock-trading

Description

This repository implements a reinforcement learning (RL) solution for stock trading focused on data-scarce environments. The implementation centers on a Q-learning agent interacting with a simulated StockTrading environment. The goal is to demonstrate that RL approaches can learn effective trading policies even when large labeled datasets are not available, by learning through interaction and reward feedback.

Key components included

- StockTrading environment: a simplified simulator that processes market history (e.g., via yfinance), executes actions (buy/sell/hold), and returns rewards based on portfolio value changes.
- QLearningAgent: a discrete Q-learning agent using a discretized state space to make the method robust to limited data.
- Training & evaluation notebooks and scripts: end-to-end Jupyter notebooks (notebooks/) and a standalone script (`src/rl_stock_trading.py`) for reproducible training and evaluation.
- Visualizations and assets: plots showing learning curves, action distributions, and policy performance stored under `assets/`.

Novelty & rationale

- Adaptability to data scarcity: the agent learns through interaction inside the environment instead of relying on large labeled datasets, making it more applicable where historical data is limited.
- Interactive learning and uncertainty handling: exploitation/exploration trade-offs allow the agent to handle uncertain market behavior.
- Simplified state representation: discretized state space reduces the need for heavy feature engineering when data are limited.

Usage & quick start

See `README.md` for quick start instructions, how to run the reorganization script, and validation steps.
