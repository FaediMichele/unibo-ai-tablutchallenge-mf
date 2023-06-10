from games.tablut.game import State, Action, Game
from typing import Callable, Any
from collections import defaultdict
import tensorflow as tf

SAVE_ITERATIONS = [1, 10, 25, 50, 100, 300, 500, 1000, 2000, 5000, 10000, 15000, 20000, 35000, 50000, 75000, 100000]
PREVIOUS_STATE_TO_MODEL = 7
BOARD_SIZE = 9


def state_hashable(state: State) -> tuple:
    player, board, turn = state
    return player, tuple(tuple(line) for line in board), turn

def unhash_state(state) -> list:
    player, board, turn = state
    return player, list(list(line) for line in board), turn


def argmax(elements: list[Any], key: Callable[[Any], float]=lambda x:x) -> int:
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
    p = []
    for a in actions:
        if a[0] == a[2]:
            p.append(matrix[a[0], a[1], a[3]])
        else:
            p.append(matrix[a[0], a[1], a[2] + BOARD_SIZE])
    return p

def policy_to_policy_matrix(policy: list[float], actions: list[Action]
                            ) -> tf.Tensor:
    action_to_policy = defaultdict(lambda: 0.0, ((a, p) for a, p in zip(actions, policy)))
    matrix = tf.constant([
        [
            [
                [action_to_policy[(x1, y1, x1, k)] for k in range(9)] +
                [action_to_policy[(x1, y1, k, y1)] for k in range(9)]
            ] for y1 in range(BOARD_SIZE)
        ] for x1 in range(BOARD_SIZE)
    ])
    return matrix

def actions_to_indexes(actions: list[Action]) -> list[tuple[int, int, int]]:
    res = []
    for a in actions:
        if a[0] == a[2]:
            res.append((a[0], a[1], a[3]))
        else:
            res.append((a[0], a[1], a[2] + BOARD_SIZE))
    return res

def no_repetition_actions(state_tensor: tf.Tensor) -> list[tuple[int,int,int]]:
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

def boolean_mask_from_coordinates(batch_action_list: list[list[tuple[int,int,int]]]) -> tf.Tensor:
    batch_size = len(batch_action_list)
    mask = tf.zeros((batch_size, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE * 2), dtype=tf.bool)
    mask = [
        [[[False for _ in range(BOARD_SIZE * 2)] for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for _ in range(batch_size)
    ]
    for batch_idx, coordinates in enumerate(batch_action_list):
        for coord in coordinates:
            mask[batch_idx][coord[0]][coord[1]][coord[2]] = True
    
    return tf.convert_to_tensor(mask, dtype=tf.bool)