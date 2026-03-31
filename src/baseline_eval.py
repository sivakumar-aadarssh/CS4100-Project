import numpy as np                                        
import random                                            
from tetris_gymnasium.envs.tetris import Tetris  

# ENVIRONMENT - using tetris-gymnasium (pip install tetris-gymnasium)

# tetris-gymnasium has 8 possible actions
num_of_actions = 8 

# 0 - move left
# 1 - move right
# 2 - move down
# 3 - rotate clockwise
# 4 - rotate counterclockwise
# 5 - hard drop (instantly drops piece to bottom)
# 6 - swap pieces
# 7 - do nothing

# creates and returns the tetris environment
def make_env():                                           
    return Tetris()

# RANDOM AGENT - an agent that, given the current state, selects an action by returning a random number between 0 and 7.
class random_agent:                                        
    def select_action(self, state):                       
        return random.randint(0, num_of_actions - 1)         


# EVALUATION PIPELINE - tracks avg score, lines cleared, survival time

# run the agent through multiple games and collect metrics
def run_episodes(agent, env, n_episodes=200):             
    scores = []
    lines_per_game = [] 
    survival_times = []   

    # loops through each episode, resetting the environment and initializing tracking variables for score, 
    # lines cleared, and steps.
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_score = 0
        total_lines = 0
        steps = 0

        # continues taking actions chosen by the agent until the episode ends, updating the observation, reward, and 
        # tracking totals each step.
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_score += reward
            total_lines += info.get('lines_cleared', 0)
            steps += 1

        scores.append(total_score)
        lines_per_game.append(total_lines)
        survival_times.append(steps)

    return {
        "avg_score": np.mean(scores),
        "avg_lines": np.mean(lines_per_game),
        "avg_survival": np.mean(survival_times),
        "scores": scores,
        "lines": lines_per_game,
        "survival": survival_times,
    }

def print_results(label, results):
    print(f"\n{label}")
    print(f" Avg Score:    {results['avg_score']:.2f}")
    print(f" Avg Lines:    {results['avg_lines']:.2f}")
    print(f" Avg Survival: {results['avg_survival']:.2f} steps")


if __name__ == "__main__":
    episodes = 200                             
    env = make_env()                    
    agent = random_agent()               
    results = run_episodes(agent, env, n_episodes=episodes)    
    print_results("Random Agent", results)