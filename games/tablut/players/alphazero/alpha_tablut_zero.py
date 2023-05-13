
import tensorflow as tf
import random
import math
import logging
from collections.abc import Callable
from tqdm import tqdm
import datetime
from collections import defaultdict

from games.board import Board
from games.game import Game, State, Action
from games.player import Player
from .util import state_hashable, argmax, unhash_state, policy_matrix_to_policy, PREVIOUS_STATE_TO_MODEL
from .model import ModelUtil
from .memory import TransferableMemory



class AlphaTablutZero(Player):
    ''' Player that take random actions '''

    def __init__(self, make_move: Callable[[None | State |Action], 
                                           None],
                                           board: Board,
                                           game: Game,
                                           player: int,
                                           greedy: bool=False,
                                           memory: TransferableMemory=None,
                                           ms_for_search: int=5000,
                                           max_moves_for_game: int=-1):
        """Create a new player tha play randomly

        Keyword arguments:
        make_move -- function that execute the next action
        board -- the board of the game. Represent the state of the game.
        game -- the game rules
        player -- the the player identification. Can be a number or anything else
        """
        self.make_move = make_move
        self.board = board
        self.game = game
        self.player = player
        self.ms_for_search = ms_for_search
        self.cache = []
        self.memory = memory if memory is not None else TransferableMemory()
        self.greedy = greedy
        self.max_moves_for_game = max_moves_for_game
        self.tmp_dict = []
        self.model = ModelUtil.load_model()
        super(Player, self).__init__()

    def next_action(self, last_action: Action, state_history: list[State]):
        ''' Function called when the opponent take the move and now is the
        turn of this player

        Keyword arguments:
        last_action -- the last move that the opponent have take
        '''
        state = state_hashable(self.board.state)
        self.memory.remove_old_branch(state)
        policy = self._mcts(state, state_history, self.ms_for_search)
        if self.greedy:
            self.make_move(self.memory.state_actions[state][argmax(policy)])
        else:
            self.make_move(random.choices(self.memory.state_actions[state],
                                          policy)[0])

    def end(self, last_action: Action, winning: str):
        """Called when a player wins.

        last_action -- the winning move
        winning -- the winning player
        """
        logging.info(f'Calling win on {Player}, winning: {winning}')
        for evaluation in self.cache:
            evaluation.append(1.0 if winning == self.player else -1.0)
        self.memory.cache.extend(self.cache)

    def _mcts(self, root: State, state_history: list[State],
              ms_for_search: int, temperature: float=1.0) ->list[float]:
        root = state_hashable(root)
        self.memory.state_actions[root] = tuple(self.game.actions(root))

        p_0, v_0, root_tensor = self._evaluate_state(
            root, self.memory.state_actions[root], state_history,
            return_state_transformed=True)

        self._expand(root, self.memory.state_actions[root], p_0)
        
        start_timer = datetime.datetime.now()
        while (datetime.datetime.now() - start_timer
               ).total_seconds() * 1000 < ms_for_search:
            s = root
            non_completed_branch = self._non_completed_branch(s)
            tree_path = []

            # condition false only when the full tree is explored
            while len(non_completed_branch) > 0:
                a_star = argmax(non_completed_branch,
                                key=lambda a: self.memory.Q[(s, a)] + 
                                self._upper_confidence_bound(s, a))
                self.tmp_dict.append((s, a_star))
                child = self.memory.parent_child[(s, a_star)]

                if child not in self.memory.state_actions.keys():
                    self.memory.N[(s, a_star)] = self.memory.N[(s, a_star)] + 1
                    tree_path.append((s, a_star))
                    s = child
                    
                    break

                non_completed_branch = self._non_completed_branch(child)
                if len(non_completed_branch) > 0:
                    self.memory.N[(s, a_star)] = self.memory.N[(s, a_star)] + 1
                    tree_path.append((s, a_star))
                    s = child
                    

                    if s not in self.memory.state_actions.keys():
                        break
                else:
                    self.memory.explored_branch[child] = True
                    if s == root:
                        # the tree is all explored
                        break

            if s == root:
                # if the tree is all explored it just take the best policy
                print("TREE ALL EXPLORED")
                break

            if self.max_moves_for_game > 0 and len(state_history) > self.max_moves_for_game:
                self.memory.state_actions[s] = tuple()
            else:
                self.memory.state_actions[s] = tuple(self.game.actions(s))
            p, v = self._evaluate_state(s, self.memory.state_actions[s],
                                        state_history + [s for s,a in tree_path])
            self._expand(s, self.memory.state_actions[s], p)
            self._backup(tree_path, v)

        count_action_taken = [self.memory.N[(root, a)] ** (1 / temperature)
                              for a in self.memory.state_actions[root]]
        denominator_policy = sum(count_action_taken)
        policy = [t / denominator_policy for t in count_action_taken]
        self.cache.append([
            root_tensor,
            policy,
            self.memory.state_actions[root]
        ])
        return policy
    
    def _non_completed_branch(self, state: State) -> list[Action]:
        return [a for a in self.memory.state_actions[state]
                if not self.memory.explored_branch[
                    self.memory.parent_child[(state, a)]]]
    


    def _upper_confidence_bound(self, state: State, action: Action,
                                cput: float = 5.0) -> float:
        return cput * self.memory.P[(state, action)] * \
            math.sqrt(sum([self.memory.N[(state, a)] for a in
                           self.memory.state_actions[state]])) /\
                            (1 + self.memory.N[(state, action)])

    def _expand(self, state: State, actions: list[Action], policy: list[float]):
        for a, pa in zip(actions, policy):
            new_state = state_hashable(self.game.result(unhash_state(state), a))
            self.memory.parent_child[(state, a)] = new_state
            self.memory.P[(state, a)] = pa
            self.memory.N[(state, a)] = 0

    def _backup(self, tree_path: list[tuple[State, Action]], state_value: float):
        for state, action in reversed(tree_path):
            self.memory.W[(state, action)] = self.memory.W[(state, action)] +\
                state_value
            self.memory.Q[(state, action)] = self.memory.W[(state, action)] /\
                self.memory.N[(state, action)]

    def _evaluate_state(self, state: State, actions: list[Action],
                        state_history: list[State],
                        return_state_transformed=False) -> tuple[list[float], float]:
        if len(actions) == 0 and self.game.is_terminal(state):
            self.memory.explored_branch[state] = True
            state_value = -1 if state[0] == self.player else 1
            if return_state_transformed:
                return [], state_value, self._state_transoform(state, state_history)
            else:
                return [], state_value
        if self.max_moves_for_game > 0 and len(state_history) > self.max_moves_for_game:
            self.memory.explored_branch[state] = True
            if return_state_transformed:
                return [], -1, self._state_transoform(state, state_history)
            else:
                return [], -1

        state_transformed = self._state_transoform(state, state_history)
        state_value, policy = ModelUtil.predict(self.model,
                                                state_transformed)
        policy = policy_matrix_to_policy(policy[0], actions)

        if return_state_transformed:
            return policy, state_value, state_transformed
        else:
            return policy, state_value
    
    def _state_transoform(self, state: State, state_history: list[State]
                          ) -> tf.Tensor:
        player, board = state
        boards = [board] + [sh[1] for sh in state_history[:PREVIOUS_STATE_TO_MODEL]]
        while len(boards) <= PREVIOUS_STATE_TO_MODEL:
            boards.append(boards[-1])
        board_shape = (len(state[1]), len(state[1][0]), 1)
        
        tensor_board = tf.stack(boards, axis=2)
        tensor_state_player = (tf.ones(board_shape, tensor_board.dtype) *
                               (player * 2 - 1))
        tensor_this_player = (tf.ones(board_shape, tensor_board.dtype) *
                              (self.player * 2 - 1))
        return tf.concat([tensor_board,
                          tensor_state_player,
                          tensor_this_player], axis=2)
    






