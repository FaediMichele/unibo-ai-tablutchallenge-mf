from games.tablut.game import State, Action
from typing import Any
from collections import defaultdict
import math
import tensorflow as tf
from .util import PREVIOUS_STATE_TO_MODEL
from typing import Union

class Tree:
    '''Class that implement the state tree dedicated to Alpha Zero algorithm.
    
    Contains the N, W, Q, P values for each children'''
    def __init__(self, state: State,
                 parent_action: tuple[Any, Action]=None) -> None:
        assert parent_action is None or isinstance(parent_action[0], Tree), parent_action
        self.state = state
        self.N: dict[Action, int] = defaultdict(lambda: 0.0001)
        self.W: dict[Action, float] = defaultdict(lambda: 0.0)
        self.Q: dict[Action, float] = defaultdict(lambda: 0.0)
        self.P: dict[Action, float] = dict()
        self.parent_child: dict[Action, Tree] = dict()
        self.actions: list[Action] = None
        self.explored_branch: bool = False
        self.parent_action: Union[tuple[Tree, Action], None] = parent_action
    
    def expand(self, child_state: State, action: Action, p: float):
        '''Add a new child with the probability given by the policy'''
        child_tree = Tree(child_state, (self, action))
        self.parent_child[action] = child_tree
        self.P[action] = p

    def backup(self, value: float):
        ''' update the W and Q value for the ancestor starting from a leaf'''
        if self.parent_action is not None:
            parent, action = self.parent_action
            parent.W[action] += value
            parent.Q[action] = parent.W[action] / parent.N[action]
            parent.backup(-value)

    def upper_confidence_bound(self, action: Action, cput: float = 5.0):
        '''Upper confidence bound function'''
        return cput * self.P[action] * \
            math.sqrt(sum([self.N[a] for a in self.actions])
                      ) / (1 + self.N[action])

    def transform(self, state_history: list[State]) -> tf.Tensor:
        '''Transform a state into a tensor for model prediction'''
        player, board, _ = self.state
        boards = [sh[1]for sh in
                  state_history[-PREVIOUS_STATE_TO_MODEL - 1: -1]] + \
                  [s[1] for s in self.get_parents_state()]
        boards = boards[-PREVIOUS_STATE_TO_MODEL - 1:]
        while len(boards) <= PREVIOUS_STATE_TO_MODEL:
            boards.insert(0, boards[0])

        boards = list(reversed(boards))

        board_shape = (len(board), len(board[0]), 1)
        
        tensor_board = tf.stack(boards, axis=2)
        tensor_state_player = tf.zeros(board_shape, tensor_board.dtype) + player * 2 - 1
        return tf.concat([tensor_board, tensor_state_player], axis=2)
    
    def get_parents_state(self) -> list[State]:
        '''Get the list of states of the ancestor of a node'''
        if self.parent_action is None:
            return [self.state]
        return self.parent_action[0].get_parents_state() + [self.state]
    
