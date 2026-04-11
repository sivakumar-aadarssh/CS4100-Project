import os
import math
import random
from dataclasses import dataclass, asdict
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

from state import extract_features
from logger import TetrisLogger


# =========================================================
# CONFIG
# =========================================================

@dataclass
class DQNConfig:
    # training
    seed: int = 42
    episodes: int = 5000
    max_steps_per_episode: int = 5000
    batch_size: int = 512
    buffer_size: int = 200_000
    min_replay_size: int = 10_000
    gamma: float = 0.99
    lr: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 10.0

    # epsilon-greedy
    eps_start: float = 1.0
    eps_end: float = 0.10
    eps_decay_steps: int = 400_000

    # target network
    target_update_every: int = 1_000

    # reward shaping
    survival_bonus: float = 0.0
    line_clear_rewards: tuple = (0.0, 40.0, 100.0, 300.0, 1200.0)
    hole_increase_penalty: float = 8.0
    hole_reduction_bonus: float = 0.0
    bumpiness_increase_penalty: float = 1.5
    height_increase_penalty: float = 2.0
    height_reduction_bonus: float = 0.0
    max_height_penalty: float = 0.5
    do_nothing_penalty: float = 2.0
    game_over_penalty: float = 200.0

    # save / eval
    model_path: str = "tetris_dqn.pt"
    plot_path: str = "training_curves.png"
    eval_episodes: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# ENV + STATE HANDLING
# =========================================================

NUM_ACTIONS = 8  # from your baseline file


def make_raw_env():
    return Tetris()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_board_from_observation(obs):
    """
    Tries to find the board/matrix inside the Tetris observation.
    """
    if isinstance(obs, dict):
        preferred_keys = ["board", "matrix", "grid", "playfield", "field"]
        for key in preferred_keys:
            if key in obs:
                arr = np.asarray(obs[key])
                if arr.ndim >= 2:
                    return arr

        # fallback: first 2D array in the dict
        for value in obs.values():
            arr = np.asarray(value)
            if arr.ndim >= 2:
                return arr

    arr = np.asarray(obs)

    # If observation is already the handcrafted feature vector, return None.
    if arr.ndim == 1 and arr.shape[0] == 13:
        return None

    if arr.ndim >= 2:
        return arr

    raise ValueError(
        "Could not extract a Tetris board from observation. "
        "Print env.reset()[0] once and inspect its structure."
    )


def observation_to_features(obs):
    """
    Converts raw environment observation -> handcrafted feature vector:
    [10 column heights, holes, bumpiness, max_height]
    """
    arr = np.asarray(obs)

    # If obs already looks like feature vector, use it directly.
    if arr.ndim == 1 and arr.shape[0] == 13:
        return arr.astype(np.float32)

    board = extract_board_from_observation(obs)
    if board is None:
        raise ValueError("Observation could not be converted to board features.")

    # Convert possible channel-last or channel-first board to 2D
    board = np.asarray(board)
    if board.ndim == 3:
        # Prefer the first channel if multi-channel
        board = board[..., 0] if board.shape[-1] <= 4 else board[0]

    # Reduce non-binary boards to occupancy
    if board.dtype != np.int32 and board.dtype != np.int64 and board.dtype != np.float32:
        board = board.astype(np.float32)

    board = (board > 0).astype(np.int32)
    return extract_features(board)


class FeatureRewardWrapper(gym.Wrapper):
    """
    Converts the environment observation to a compact feature vector and
    replaces the environment reward with a shaped reward designed for DQN.
    """

    def __init__(self, env, config: DQNConfig):
        super().__init__(env)
        self.config = config
        self.prev_features = None

        # 10 heights + holes + bumpiness + max_height = 13
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        features = observation_to_features(obs)
        self.prev_features = features.copy()
        return features, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        features = observation_to_features(obs)
        done = terminated or truncated

        reward = self.compute_shaped_reward(
            prev_features=self.prev_features,
            curr_features=features,
            lines_cleared=info.get("lines_cleared", 0),
            done=done,
            action=action,
        )

        self.prev_features = features.copy()
        return features, reward, terminated, truncated, info

    def compute_shaped_reward(self, prev_features, curr_features, lines_cleared, done, action=None):
        """
        Reward design:
        + strongly reward line clears
        - strongly penalize making more holes
        - penalize increasing bumpiness / stack height
        - penalize tall stacks in general
        - penalize doing nothing
        - large game over penalty
        """
        prev_holes = float(prev_features[-3])
        prev_bumpiness = float(prev_features[-2])
        prev_max_height = float(prev_features[-1])

        curr_holes = float(curr_features[-3])
        curr_bumpiness = float(curr_features[-2])
        curr_max_height = float(curr_features[-1])

        reward = self.config.survival_bonus

        # line clear reward
        if 0 <= lines_cleared < len(self.config.line_clear_rewards):
            reward += self.config.line_clear_rewards[lines_cleared]
        else:
            reward += 1200.0

        # holes
        hole_delta = curr_holes - prev_holes
        if hole_delta > 0:
            reward -= self.config.hole_increase_penalty * hole_delta
        else:
            reward += self.config.hole_reduction_bonus * abs(hole_delta)

        # bumpiness
        bump_delta = curr_bumpiness - prev_bumpiness
        if bump_delta > 0:
            reward -= self.config.bumpiness_increase_penalty * bump_delta

        # stack height
        height_delta = curr_max_height - prev_max_height
        if height_delta > 0:
            reward -= self.config.height_increase_penalty * height_delta
        else:
            reward += self.config.height_reduction_bonus * abs(height_delta)

        # penalize tall stacks in general
        reward -= self.config.max_height_penalty * curr_max_height

        # discourage do nothing
        if action == 7:
            reward -= self.config.do_nothing_penalty

        # game over
        if done:
            reward -= self.config.game_over_penalty

        return float(reward)


# =========================================================
# REPLAY BUFFER
# =========================================================

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            Transition(state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# =========================================================
# MODEL
# =========================================================

class QNetwork(nn.Module):
    def __init__(self, input_dim=13, num_actions=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# AGENT
# =========================================================

class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.replay = ReplayBuffer(config.buffer_size)
        self.train_steps = 0

    def epsilon(self, global_step):
        frac = min(1.0, global_step / self.config.eps_decay_steps)
        return self.config.eps_start + frac * (self.config.eps_end - self.config.eps_start)

    def select_action(self, state, global_step, greedy=False):
        if (not greedy) and random.random() < self.epsilon(global_step):
            return random.randint(0, NUM_ACTIONS - 1)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self):
        if len(self.replay) < self.config.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.config.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s,a)
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Double DQN target:
        # action selected by policy net, value from target net
        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(next_states), dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path):
        payload = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "config": asdict(self.config),
        }
        torch.save(payload, path)

    def load(self, path):
        payload = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(payload["policy_net"])
        self.target_net.load_state_dict(payload["target_net"])


# =========================================================
# TRAIN / EVAL
# =========================================================

def evaluate_agent(agent: DQNAgent, config: DQNConfig, n_episodes=50):
    env = FeatureRewardWrapper(make_raw_env(), config)

    rewards = []
    lines = []
    survivals = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        total_lines = 0
        steps = 0

        while not done and steps < config.max_steps_per_episode:
            action = agent.select_action(state, global_step=10**12, greedy=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # temporarily print the first few info dictionaries to verify the line-clear key
            if episode == 0 and steps < 3:
                print("Eval info keys:", info.keys())
                print("Eval info:", info)

            total_reward += reward
            total_lines += info.get("lines_cleared", 0)
            steps += 1
            state = next_state

        rewards.append(total_reward)
        lines.append(total_lines)
        survivals.append(steps)

    env.close()

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_lines": float(np.mean(lines)),
        "avg_survival": float(np.mean(survivals)),
        "rewards": rewards,
        "lines": lines,
        "survival": survivals,
    }


def train_dqn(config: DQNConfig):
    set_seed(config.seed)

    env = FeatureRewardWrapper(make_raw_env(), config)
    logger = TetrisLogger()
    agent = DQNAgent(config)

    global_step = 0
    best_avg_lines = -float("inf")

    print("Starting training...")
    print(f"Using device: {config.device}")
    print(f"Config: {config}")

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset()
        done = False

        episode_reward = 0.0
        episode_lines = 0
        episode_steps = 0
        losses = []

        while not done and episode_steps < config.max_steps_per_episode:
            action = agent.select_action(state, global_step=global_step, greedy=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # temporarily print the first few info dictionaries to verify the line-clear key
            if episode == 1 and episode_steps < 5:
                print("Train info keys:", info.keys())
                print("Train info:", info)

            agent.replay.add(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            episode_reward += reward
            episode_lines += info.get("lines_cleared", 0)
            episode_steps += 1
            global_step += 1
            state = next_state

        logger.log(episode_reward, episode_lines, episode_steps)

        if episode % 25 == 0:
            avg_reward = np.mean(logger.episode_rewards[-25:])
            avg_lines = np.mean(logger.lines_cleared[-25:])
            avg_survival = np.mean(logger.survival_times[-25:])
            avg_loss = np.mean(losses) if losses else float("nan")
            eps = agent.epsilon(global_step)

            print(
                f"Episode {episode:4d} | "
                f"avg_reward={avg_reward:8.2f} | "
                f"avg_lines={avg_lines:6.2f} | "
                f"avg_survival={avg_survival:7.2f} | "
                f"avg_loss={avg_loss:8.4f} | "
                f"epsilon={eps:.3f}"
            )

            # save best by recent avg lines
            if avg_lines > best_avg_lines:
                best_avg_lines = avg_lines
                agent.save(config.model_path)
                print(f"Saved improved model to {config.model_path}")

    logger.plot(config.plot_path)
    env.close()

    # load best saved model before final eval
    if os.path.exists(config.model_path):
        agent.load(config.model_path)

    results = evaluate_agent(agent, config, n_episodes=config.eval_episodes)
    print("\nFinal Evaluation")
    print(f" Avg Reward:   {results['avg_reward']:.2f}")
    print(f" Avg Lines:    {results['avg_lines']:.2f}")
    print(f" Avg Survival: {results['avg_survival']:.2f}")

    return agent, logger, results


# =========================================================
#     HYPERPARAMETER ITERATION HELPER
# =========================================================

def run_reward_experiment(
    hole_penalty,
    line_clear_rewards,
    game_over_penalty,
    episodes=500,
    model_path="temp_tetris_dqn.pt"
):
    """
    Small helper for reward-function iteration.
    Lets you quickly compare reward settings without rewriting the config.
    """
    cfg = DQNConfig(
        episodes=episodes,
        hole_increase_penalty=hole_penalty,
        line_clear_rewards=line_clear_rewards,
        game_over_penalty=game_over_penalty,
        model_path=model_path,
        plot_path="temp_training_curves.png",
    )
    _, _, results = train_dqn(cfg)
    return results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    config = DQNConfig(
        episodes=5000,
        batch_size=512,
        buffer_size=200_000,
        min_replay_size=10_000,
        gamma=0.99,
        lr=5e-4,
        eps_start=1.0,
        eps_end=0.10,
        eps_decay_steps=400_000,
        target_update_every=1_000,

        # reward design
        survival_bonus=0.0,
        line_clear_rewards=(0.0, 40.0, 100.0, 300.0, 1200.0),
        hole_increase_penalty=8.0,
        hole_reduction_bonus=0.0,
        bumpiness_increase_penalty=1.5,
        height_increase_penalty=2.0,
        height_reduction_bonus=0.0,
        max_height_penalty=0.5,
        do_nothing_penalty=2.0,
        game_over_penalty=200.0,

        model_path="tetris_dqn.pt",
        plot_path="training_curves.png",
        eval_episodes=50,
    )

    train_dqn(config)
