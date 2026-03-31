"""
MDP Formulation for Tetris RL Agent
-------------------------------------
States: Hand-crafted feature vector — column heights, holes, bumpiness, max height
Actions: Discrete — (column, rotation) pairs for piece placement
Transition: Deterministic given action; next piece drawn randomly (stochasticity)
Reward: Shaped reward based on lines cleared, holes created, and bumpiness
"""

class TetrisMDP:
    def __init__(self):
        self.discount_factor = 0.99  # gamma

    def compute_reward(self, lines_cleared, holes, bumpiness, game_over):
        """
        Reward structure:
        - Large reward for clearing lines (exponential to encourage multi-clears)
        - Penalty for holes (blocks access to empty space below)
        - Penalty for bumpiness (uneven surface is hard to fill)
        - Large penalty for game over
        """
        if game_over:
            return -100

        line_reward = [0, 100, 300, 500, 800][lines_cleared]
        hole_penalty = -10 * holes
        bumpiness_penalty = -2 * bumpiness

        return line_reward + hole_penalty + bumpiness_penalty

    def describe(self):
        print("=== Tetris MDP ===")
        print("States: Feature vector [col heights x10, holes, bumpiness, max_height]")
        print("Actions: (column, rotation) placement pairs")
        print("Transition: Deterministic placement + random next piece")
        print("Reward: Lines cleared - hole penalty - bumpiness penalty")
        print(f"Discount factor (gamma): {self.discount_factor}")