# 🐦 Flappy Bird — DQN Agent

A Deep Reinforcement Learning agent trained to play Flappy Bird using **Deep Q-Network (DQN)** with Experience Replay and a Target Network — built with PyTorch and `flappy_bird_gymnasium`.

---

## Project Structure

```
DQN_FlappyBird/
├── agent.py               # Training loop — ε-greedy, learning, target sync
├── dqn.py                 # Policy & Target network definition
├── experience_replay.py   # Replay memory buffer
├── game_flappy_bird.py    # Environment setup & interaction
├── parameters.yaml        # Hyperparameters
└── runs/
    ├── flappybirdv0.pt    # Saved model weights
    └── flappybirdv0.log   # Training logs
```

---

## How It Works

- The agent observes 12 features (pipe positions, bird position, velocity, rotation)
- Chooses to **flap** or **do nothing** using an ε-greedy policy
- Transitions are stored in a **Replay Memory** and sampled randomly for training
- A **Target Network** is periodically synced for stable Q-value updates

---

## Setup & Run

```bash
pip install flappy-bird-gymnasium gymnasium torch pyyaml
python agent.py
```

---

## Dependencies

```
torch · flappy-bird-gymnasium · gymnasium · pyyaml
```

---

## Author

**Deepanshu Tevathiya** — AI / ML Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/deepanshu-tevathiya/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/DeepanshuTevathiya)
