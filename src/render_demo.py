import time
from tetris_gymnasium.envs.tetris import Tetris
from state import extract_features
from tetris_dqn import DQNAgent, DQNConfig, FeatureRewardWrapper, make_raw_env
import torch

def run_visual_demo(model_path="tetris_dqn.pt"):
    config = DQNConfig()
    
    # Make environment with rendering
    raw_env = Tetris(render_mode="human")
    env = FeatureRewardWrapper(raw_env, config)
    
    # Load trained agent
    agent = DQNAgent(config)
    agent.load(model_path)
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("Running visual demo...")
    time.sleep(3)  # gives you time to click the window
    while not done and steps < 1000:
        env.render()
        action = agent.select_action(state, global_step=10**12, greedy=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        time.sleep(0.05)  # slow it down so it's watchable

    print(f"Game over! Steps: {steps}, Total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    run_visual_demo()