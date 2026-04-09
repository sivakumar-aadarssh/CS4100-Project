# CS4100-Project
CS4100 AI Project

## Overview
This script renders a live visualization of a trained DQN agent playing Tetris. It loads the trained model from `tetris_dqn.pt`, wraps the Tetris environment with `FeatureRewardWrapper` to handle feature extraction and reward shaping, and displays each frame in a pygame window at 10 fps. The agent plays greedily — always picking the highest Q-value action with no random exploration. The window stays open after the game ends until manually closed.

## Key Components

**Environment Setup:** The raw Tetris environment is initialized with `render_mode="rgb_array"`, which returns each frame as a numpy array instead of directly opening a window. It is then wrapped with `FeatureRewardWrapper`, which converts the raw board observation into a 21-feature vector (10 column heights, holes, bumpiness, and max height) and replaces the environment's default reward with a shaped reward designed for DQN training.

**Agent:** The `DQNAgent` is initialized with the default `DQNConfig` and loaded from the saved checkpoint `tetris_dqn.pt`. During the demo it runs in greedy mode, meaning it always selects the action with the highest Q-value from the policy network with no random exploration.

**Game Loop:** The agent steps through the environment for up to 1000 steps, rendering each frame and tracking total reward and steps taken. Once the game ends, the final score is printed and the window remains open until manually closed.

## Requirements
- `tetris-gymnasium`
- `pygame`
- `numpy`
- `torch`
- A trained model checkpoint (`tetris_dqn.pt`)

## Usage
```bash
python render_demo.py
```
