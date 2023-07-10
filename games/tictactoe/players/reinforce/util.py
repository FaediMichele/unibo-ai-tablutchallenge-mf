from games.tictactoe.game import State, Action, Game
from typing import Callable, Any
from collections import defaultdict
import tensorflow as tf

SAVE_ITERATIONS = [1, 10, 25, 50, 100, 300, 500, 1000, 2000, 5000, 10000, 15000, 20000, 35000, 50000, 75000, 100000]
PREVIOUS_STATE_TO_MODEL = 7
BOARD_SIZE = 9



def state_hashable(state: State) -> tuple:
    """Convert a state into an hashable version (convert the board from list of list to tuple of tuple)"""
    player, board, turn = state
    return player, tuple(tuple(line) for line in board), turn

def unhash_state(state) -> list:
    """Convert an hashable state into the normal version (convert the board from tuple of tuple to list of list)"""
    player, board, turn = state
    return player, list(list(line) for line in board), turn


def argmax(elements: list[Any], key: Callable[[Any], float]=lambda x:x) -> int:
    """Argmax function using a key as a fuction to apply for compunting the value"""
    if len(elements) == 0:
        raise IndexError('Empty argmax list')
    k = 0
    k_val = key(elements[k])
    for i in range(1, len(elements)):
        i_val = key(elements[i])
        if i_val > k_val:
            k = i
            k_val = i_val
    return elements[k]

def policy_matrix_to_policy(matrix, actions: list[Action]) -> list[float]:
    """Convert from the policy matrix (output of keras model) into a distribution"""
    p = []
    for a in actions:
        p.append(matrix[a[0], a[1]])
    return p

def actions_to_indexes(actions: list[Action]) -> list[tuple[int, int, int]]:
    """Convert list of Actions into position of probability in the matrix representation of the policy"""
    res = []
    for a in actions:
        res.append((a[0], a[1]))
    return res

def policy_to_policy_matrix(policy: list[float], actions: list[Action]
                            ) -> tf.Tensor:
    """From list of action to matrix representation of the policy"""
    action_to_policy = defaultdict(lambda: 0.0, ((a, p) for a, p in zip(actions, policy)))
    matrix = tf.constant([
        [
            action_to_policy[(x1, y1)]
            for y1 in range(BOARD_SIZE)
        ] for x1 in range(BOARD_SIZE)
    ])
    return matrix

def no_repetition_actions(state_tensor: tf.Tensor) -> list[tuple[int,int,int]]:
    """Return the action indices that leads to repeated states"""
    actual_board = state_tensor[:, : , 0].numpy().tolist()
    m1_board = state_tensor[:, :, 1].numpy().tolist()
    m3_board = state_tensor[:,:, 5].numpy().tolist()
    player = int(float(state_tensor[0, 0, -1].numpy()) / 2 + 0.5) + 1 % 2
    actual_game_state = player, actual_board, -10
    game = Game()
    actions = game.actions(actual_game_state)

    actions_to_discourage = []
    def equal_state(b1, b2):
        for x in range(len(b1)):
            for y in range(len(b1[0])):
                if b1[x][y] != b2[x][y]:
                    return False
        return True
    
    for a in actions:
        new_state = game.result(actual_game_state, a)
        if equal_state(new_state[1], m1_board) or equal_state(new_state[1], m3_board):
            actions_to_discourage.append(a)
            
    ret =  actions_to_indexes(actions_to_discourage)
    return ret

def correct_actions(state_tensor: tf.Tensor) -> list[tuple[int,int,int]]:
    """Get the indexes of actions possible in the given state. The idea is to demote the action not possible"""
    actual_board = state_tensor[:, : , 0].numpy().tolist()
    player = int(float(state_tensor[0, 0, -1].numpy()) / 2 + 0.5) + 1 % 2
    actual_game_state = player, actual_board, -10
    game = Game()
    actions = game.actions(actual_game_state)
    return actions_to_indexes(actions)

def boolean_mask_from_coordinates(batch_action_list: list[list[tuple[int,int]]], positives=True) -> tf.Tensor:
    """Create a boolean mask of the policy matrix given a list of action for each batch"""
    batch_size = len(batch_action_list)
    if positives:
        mask = tf.zeros((batch_size, BOARD_SIZE, BOARD_SIZE), dtype=tf.bool)
    else:
        mask = tf.ones((batch_size, BOARD_SIZE, BOARD_SIZE), dtype=tf.bool)
    mask = [
        [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for _ in range(batch_size)
    ]
    for batch_idx, coordinates in enumerate(batch_action_list):
        for coord in coordinates:
            if positives:
                mask[batch_idx][coord[0]][coord[1]] = True
            else:
                mask[batch_idx][coord[0]][coord[1]]= False
    
    return tf.convert_to_tensor(mask, dtype=tf.bool)






    


