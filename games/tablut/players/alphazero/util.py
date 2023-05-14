from games.tablut.game import State, Action
from typing import Callable, Any
from collections import defaultdict
import tensorflow as tf

SAVE_ITERATIONS = [1, 10, 25, 50, 100, 300, 500, 1000, 2000, 5000, 10000]
PREVIOUS_STATE_TO_MODEL = 7


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
            p.append(matrix[a[0], a[1], a[2] + 9])
    return p

def policy_to_policy_matrix(policy: list[float], actions: list[Action]
                            ) -> tf.Tensor:
    action_to_policy = defaultdict(lambda: 0.0, ((a, p) for a, p in zip(actions, policy)))
    matrix = tf.constant([
        [
            [
                [action_to_policy[(x1, y1, x1, k)] for k in range(9)] +
                [action_to_policy[(x1, y1, k, y1)] for k in range(9)]
            ] for y1 in range(9)
        ] for x1 in range(9)
    ])
    return matrix


    


