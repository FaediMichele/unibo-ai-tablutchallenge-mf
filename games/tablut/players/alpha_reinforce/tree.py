from games.tablut.game import State, Action
from typing import Any
from collections import defaultdict
import math
import tensorflow as tf
from .util import PREVIOUS_STATE_TO_MODEL

class Tree:
    def __init__(self, player: int, state: State,
                 parent_action: tuple[Any, Action]=None) -> None:
        assert parent_action is None or isinstance(parent_action[0], Tree)
        self.state = state
        self.player = player
        self.N: dict[Action, int] = defaultdict(lambda: 0)
        self.W: dict[Action, float] = defaultdict(lambda: 0.0)
        self.Q: dict[Action, float] = defaultdict(lambda: 0.0)
        self.P: dict[Action, float] = dict()
        self.parent_child: dict[Action, Tree] = dict()
        self.actions: list[Action] = None
        self.explored_branch: bool = False
        self.parent_action: tuple[Tree, Action] | None = parent_action
    
    def expand(self, child_state: State, action: Action, p: float):
        child_tree = Tree(self.player, child_state, (self, action))
        self.parent_child[action] = child_tree
        self.P[action] = p

    def backup(self, value: float, invert: bool=False):
        if self.parent_action is not None:
            parent, action = self.parent_action
            parent.W[action] += value
            parent.Q[action] = parent.W[action] / parent.N[action]
            parent.backup(-value if invert else value)

    def upper_confidence_bound(self, action: Action, cput: float = 5.0):
        return cput * self.P[action] * \
            math.sqrt(sum([self.N[a] for a in self.actions])
                      ) / (1 + self.N[action])

    def transform(self, state_history: list[State]) -> tf.Tensor:
        player, board, _ = self.state
        boards = [s[1] for s in self.get_parents_state()] + \
            [sh[1]for sh in state_history[-PREVIOUS_STATE_TO_MODEL - 1:-1]]
        boards = boards[-PREVIOUS_STATE_TO_MODEL:]
        while len(boards) <= PREVIOUS_STATE_TO_MODEL:
            boards.insert(0, boards[0])
        board_shape = (len(board), len(board[0]), 1)
        
        tensor_board = tf.stack(boards, axis=2)
        tensor_state_player = (tf.ones(board_shape, tensor_board.dtype) *
                               (player * 2 - 1))
        return tf.concat([tensor_board, tensor_state_player], axis=2)
    
    def get_parents_state(self) -> list[State]:
        if self.parent_action is None:
            return [self.state]
        return self.parent_action[0].get_parents_state() + [self.state]
    
    def get_non_completed_branch(self) -> list[Action]:
        non_complete_actions = [a for a in self.actions
                if not self.parent_child[a].explored_branch]
        if len(non_complete_actions) == 0:
            self.explored_branch = True
        return non_complete_actions