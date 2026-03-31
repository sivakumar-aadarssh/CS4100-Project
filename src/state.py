import numpy as np

def get_board_heights(board):
    heights = []
    for col in range(board.shape[1]):
        col_data = board[:, col]
        filled = np.where(col_data > 0)[0]
        heights.append(board.shape[0] - filled[0] if len(filled) > 0 else 0)
    return heights

def get_holes(board, heights):
    holes = 0
    for col in range(board.shape[1]):
        top = board.shape[0] - heights[col]
        for row in range(top + 1, board.shape[0]):
            if board[row, col] == 0:
                holes += 1
    return holes

def get_bumpiness(heights):
    return sum(abs(heights[i] - heights[i+1]) for i in range(len(heights) - 1))

def extract_features(board):
    """
    Hand-crafted feature vector for Tetris state representation.
    Returns: numpy array of [heights..., holes, bumpiness, max_height]
    """
    heights = get_board_heights(board)
    holes = get_holes(board, heights)
    bumpiness = get_bumpiness(heights)
    max_height = max(heights)

    features = np.array(heights + [holes, bumpiness, max_height], dtype=np.float32)
    return features