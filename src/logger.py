import matplotlib.pyplot as plt
import numpy as np

class TetrisLogger:
    def __init__(self):
        self.episode_rewards = []
        self.lines_cleared = []
        self.survival_times = []

    def log(self, reward, lines, survival):
        self.episode_rewards.append(reward)
        self.lines_cleared.append(lines)
        self.survival_times.append(survival)

    def plot(self, save_path="training_curves.png"):
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # Reward over time
        axes[0].plot(self.episode_rewards, color='royalblue', alpha=0.6)
        axes[0].plot(self._moving_avg(self.episode_rewards), color='blue', linewidth=2)
        axes[0].set_title("Reward over Episodes")
        axes[0].set_ylabel("Total Reward")

        # Lines cleared over time
        axes[1].plot(self.lines_cleared, color='mediumseagreen', alpha=0.6)
        axes[1].plot(self._moving_avg(self.lines_cleared), color='green', linewidth=2)
        axes[1].set_title("Lines Cleared per Episode")
        axes[1].set_ylabel("Lines Cleared")

        # Survival time over time
        axes[2].plot(self.survival_times, color='tomato', alpha=0.6)
        axes[2].plot(self._moving_avg(self.survival_times), color='red', linewidth=2)
        axes[2].set_title("Survival Time per Episode")
        axes[2].set_ylabel("Steps Survived")
        axes[2].set_xlabel("Episode")

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")

    def _moving_avg(self, data, window=20):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')