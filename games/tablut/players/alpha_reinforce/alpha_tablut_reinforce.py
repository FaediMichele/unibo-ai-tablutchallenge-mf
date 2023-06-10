
import tensorflow as tf
import random
import math
import logging
from collections.abc import Callable
from tqdm import tqdm
import datetime
from collections import defaultdict
import gc

from games.board import Board
from games.game import Game, State, Action
from games.player import Player
from .util import state_hashable, argmax, unhash_state, policy_matrix_to_policy, PREVIOUS_STATE_TO_MODEL
from .tree import Tree
from .model import Model

class AlphaTablutReinforce(Player):
    ''' Player that take random actions '''

    def __init__(self, make_move: Callable[[None | State |Action], None],
                 board: Board,
                 game: Game,
                 player: int,
                 model: Model,
                 training: bool=False,
                 ms_for_search: int=5000,
                 node_for_search: int=None):
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
        self.node_for_search = node_for_search
        self.cache: list[tuple[tf.Tensor, list[float], list[Action], Action]] = []
        self.tree: Tree = None
        self.training = training
        self.model = model
        super(Player, self).__init__()

    def next_action(self, last_action: Action, state_history: list[State]):
        ''' Function called when the opponent take the move and now is the
        turn of this player

        Keyword arguments:
        last_action -- the last move that the opponent have take
        '''
        if self.tree is None:
            self.tree = Tree(self.player, self.board.state)
        else:
            if last_action in self.tree.parent_child:
                self.tree = self.tree.parent_child[last_action]
                self.tree.parent = None
            else:
                self.tree = Tree(self.player, self.board.state)
        gc.collect()
        policy = self._mcts(self.tree, state_history, self.ms_for_search,
                            training=self.training)
        if self.training:
            # the if is true when the model assign 0 probability to all
            # possible moves
            if sum(policy) < 0.001:
                action_taken = random.choice(self.tree.actions)[0]
            else:
                action_taken = random.choices(self.tree.actions, policy)[0]
            self.cache[-1].append(action_taken)
            self.make_move(action_taken)
        else:
            self.make_move(self.tree.actions[argmax(range(len(policy)),
                                                    key=lambda x: policy[x])])

    def end(self, last_action: Action, opponent: Player, winner: str):
        """Called when a player wins.

        last_action -- the winning move
        winning -- the winning player
        """
        logging.info(f'Calling win on {Player}, winning: {winner}')
        if self.training:
            if winner == 'draw':
                g = 0
            elif winner == self.player: # actually never happen
                g = 1.0
            else:
                g = -1.0
            
            for step in self.cache:
                step.append(g)
            for step in opponent.cache:
                step.append(-g)
            
            self.model.save_cache(self.cache)
            opponent.model.save_cache(opponent.cache)
            
            self.model.train_episode(self.cache + opponent.cache)

    def _mcts(self, root: Tree, state_history: list[State],
              temperature: float=1.0, training=False
              ) ->list[float]:
        
        if root.actions is None:
            root.actions = tuple(self.game.actions(root.state))

        
        p_0, v_0, _, root_tensor = self._evaluate_state(
            root, state_history, return_state_transformed=True)

        self._expand(root, p_0)

        c = 0
        if self.node_for_search is not None:
            stop_condition = lambda: c < self.node_for_search
        else:
            start_timer = datetime.datetime.now()
            stop_condition = lambda: (datetime.datetime.now() - start_timer
                                      ).total_seconds() * 1000 < self.ms_for_search
       
        
        while stop_condition():
            s = root
            tree_depth = 0
            # reached a leaf or a terminal state
            while s.actions is not None and len(s.actions) > 0:
                a_star = argmax(s.actions, key=lambda a:
                                s.Q[a] + s.upper_confidence_bound(a))
                child = s.parent_child[a_star]

                s.N[a_star] += 1
                s = child
                tree_depth += 1
            c += 1

            if s.actions is None:
                s.actions = tuple(self.game.actions(s.state))
                p, v, is_final = self._evaluate_state(s, state_history)
                self._expand(s, p)
                self._backup(s, v, is_final)
            else:
                print(f"found terminal state. {s.parent_action[0].N[a_star]}")
        count_action_taken = [root.N[a] ** (1 / temperature)
                              for a in root.actions]
        denominator_policy = sum(count_action_taken)
        policy = [(t ** (1 / temperature)) / denominator_policy for t in count_action_taken]
        if training:
            self.cache.append([
                root_tensor,
                policy,
                root.actions
            ])
        return policy

    def _expand(self, state_tree: Tree, policy: list[float]):
        for a, pa in zip(state_tree.actions, policy):
            new_state = state_hashable(self.game.result(
                unhash_state(state_tree.state), a))
            state_tree.expand(new_state, a, pa)

    def _backup(self, leaf: Tree, state_value: float, invert: bool):
        leaf.backup(state_value)

    def _evaluate_state(self, state_tree: Tree,
                        state_history: list[State],
                        return_state_transformed=False) -> tuple[list[float], float]:
        if self.game.is_terminal(state_tree.state):
            state_tree.explored_branch = True
            if state_tree.state[2] >= 0:
                state_value = 0
            elif state_tree.state[0] == self.player:
                state_value = -1
            else:
                state_value = 1

            if return_state_transformed:
                return [], state_value, True, state_tree.transform(state_history)
            else:
                return [], state_value, True

        state_transformed = state_tree.transform(state_history)
        state_value, policy = self.model.predict(state_transformed)
        policy = policy_matrix_to_policy(policy, state_tree.actions)

        if return_state_transformed:
            return policy, state_value, False, state_transformed
        else:
            return policy, state_value, False
