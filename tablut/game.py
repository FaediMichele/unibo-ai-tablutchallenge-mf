import numpy as np

# example action: (start_row_id, start_column_id, dest_row_id, dest_column_id)


black = -1
white = 1
king = 2

# Class that contains rules, take actions and states


def create_root():
    board = np.zeros((9, 9), dtype=np.int8)
    board[3:6, 0] = black
    board[4, 1] = black
    board[4, 7] = black
    board[3:6, 8] = black
    board[0, 3:6] = black
    board[8, 3:6] = black
    board[1, 4] = black
    board[7, 4] = black
    board[4, 2:7] = white
    board[2:7, 4] = white
    board[4, 4] = king
    return board


def get_piece_positions(state, player):
    return state.where(state < 0 if player == "black" else state >= 0)


def get_piece_actions(state, d, player):
    # TODO define here the game rules
    return []


def actions(state, player):
    actions = []
    for d in get_piece_positions(state, player):
        actions.append(get_piece_actions(state, d, player))


if __name__ == '__main__':
    pass
