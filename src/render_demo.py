import time
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris
from state import extract_features
import pygame
from tetris_dqn import DQNAgent, DQNConfig, FeatureRewardWrapper, make_raw_env
import torch

def run_visual_demo(model_path="tetris_dqn.pt"):
    # set up the environment and agent
    config = DQNConfig()
    raw_env = Tetris(render_mode="rgb_array")
    env = FeatureRewardWrapper(raw_env, config)

    agent = DQNAgent(config)
    agent.load(model_path)

    # set up the state and tracking variables
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    pygame.init()

    # take one step to get a real frame for window sizing
    action = agent.select_action(state, global_step=10**12, greedy=True)
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    steps += 1

    # set up the window for pygame
    frame = raw_env.render()
    screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
    pygame.display.set_caption("Tetris DQN Demo")
    clock = pygame.time.Clock()

    print("Running visual demo...")

    while not done and steps < 1000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # convert rgb_array frame to pygame surface and draw to screen
        frame = raw_env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        # agent selects and takes action
        action = agent.select_action(state, global_step=10**12, greedy=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        clock.tick(10)

    print(f"Game over! Steps: {steps}, Total reward: {total_reward:.2f}")

    # wait for user to close the pygame window
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

    pygame.quit()
    env.close()

if __name__ == "__main__":
    run_visual_demo()
