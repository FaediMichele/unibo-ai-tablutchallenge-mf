
import tensorflow as tf
import random
import math
import logging
from collections.abc import Callable
from tqdm import tqdm
import datetime
from collections import defaultdict
import gc
from typing import Union

from games.board import Board
from games.game import Game, State, Action
from games.player import Player
#from games.tablut.optimized.game import actions as opt_actions, result as opt_result, is_terminal as opt_is_terminal
from .util import state_hashable, argmax, unhash_state, policy_matrix_to_policy, PREVIOUS_STATE_TO_MODEL
from .tree import Tree
from .model import Model

class AlphaZero(Player):
    ''' AI player that play using alpha zero algorithm '''

    def __init__(self, make_move: Callable[[Union[None, State, Action]], None],
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
        self.tree = None
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
            self.tree = Tree(self.board.state)
        else:
            if last_action in self.tree.parent_child:
                self.tree = self.tree.parent_child[last_action]
                self.tree.parent_action = None
            else:
                self.tree = Tree(self.board.state)

        gc.collect()
        if len(state_history) > 10:
            temperature = 1
        else:
            temperature = 1

        policy = self._mcts(self.tree, state_history, training=self.training, temperature=temperature)
        if self.training:
            # the if is true when the model assign 0 probability to all
            # possible moves
            
            
            policy, actions = zip(*policy)
            
            if sum(policy) < 0.001:
                action_taken = random.choice(self.tree.actions)[0]
            else:
                assert actions == self.tree.actions
                action_taken = random.choices(actions, policy)[0]
            print(f"Player {'W' if self.player == 0 else 'B'} - max - mean: {max(policy) - sum(policy) / len(policy):.5f}")
            self.cache[-1].append(action_taken)
        else:
            if len(state_history) < 5: 
                temperature = 1.5
                policy, actions = zip(*policy)
                not_training_policy = [((t ** (1 / temperature)) / sum(policy)) for t in policy]
                action_taken = random.choices(actions, not_training_policy)[0]
            else:
                action_taken = self.tree.actions[argmax(range(len(policy)), key=lambda x: policy[x])]

            
        self.tree = self.tree.parent_child[action_taken]
        self.tree.parent_action = None
        self.make_move(action_taken)

    def end(self, last_action: Action, opponent: Player, winner: str):
        """Called when a player wins.

        last_action -- the winning move
        winning -- the winning player
        """
        logging.info(f'Calling win on {Player}, winning: {winner}')
        if self.training:
            if winner == 'draw':
                g = -0.1
                go = -0.1
            elif winner == self.player: # actually never happen
                g = 1.0
                go = -1.0
            else:
                g = -1.0
                go = 1.0
            print(self.player, winner, g)
            for step in self.cache:
                step.append(g)
            for step in opponent.cache:
                step.append(go)
            
            self.model.save_cache(self.cache)
            opponent.model.save_cache(opponent.cache)
            
            if self.model.remote:
                self.model.train_episode(self.cache + opponent.cache)

    def _mcts(self, root: Tree, state_history: list[State],
              temperature: float=1.0, training=False
              ) ->list[float]:
        # if new unvisited state calculate the actions
        if root.actions is None:
            root.actions = tuple(self.game.actions(root.state))
        
        # always evaluate root
        p_0, v_0, _, root_tensor = self._evaluate_state(
            root, state_history, return_state_transformed=True)

        # if the root is not evaluated 
        if len(list(root.parent_child.keys())) == 0:
            self._expand(root, p_0)

        # decide when to stop the search either by a timeout or a limit on evaluated nodes
        c = 0
        if self.node_for_search is not None:
            stop_condition = lambda: c < self.node_for_search
        else:
            start_timer = datetime.datetime.now()
            stop_condition = lambda: (datetime.datetime.now() - start_timer
                                      ).total_seconds() * 1000 < self.ms_for_search
       
        # start search
        while stop_condition():
            s = root
            tree_depth = 0
            # while not reached a leaf or a terminal state
            while s.actions is not None and len(s.actions) > 0:
                a_star = argmax(s.actions, key=lambda a:
                                s.Q[a] + s.upper_confidence_bound(a))
                child = s.parent_child[a_star]

                s.N[a_star] += 1
                s = child
                tree_depth += 1
            
            # reached a leaf
            if s.actions is None:
                s.actions = tuple(self.game.actions(s.state))
                
                p, v, is_final = self._evaluate_state(s, state_history)
                self._expand(s, p)
                self._backup(s, -v) # v negated
                c += 1
            else: # reached a leaf that is a terminal state
                c += 0.1
        print(f"\nanalized {c:.1f} states. Reached {tree_depth} depth")
        count_action_taken = [(root.N[a] ** (1 / temperature), a)
                              for a in root.actions]
        denominator_policy = sum(t for t, _ in count_action_taken)
        policy = [(((t ** (1 / temperature)) / denominator_policy), a) for t, a in count_action_taken]

        # store data to cache. At the end of the episode the data is stored to a file
        if training:
            self.cache.append([
                root_tensor,
                [prob for prob, _ in policy],
                root.actions
            ])
        return policy

    
    def _expand(self, state_tree: Tree, policy: list[float]):
        '''create new leaf starting from a node'''
        for a, pa in zip(state_tree.actions, policy):
            new_state = state_hashable(self.game.result(
                unhash_state(state_tree.state), a))
            state_tree.expand(new_state, a, pa)

    
    def _backup(self, leaf: Tree, state_value: float):
        '''begin the backup phase'''
        leaf.backup(state_value)


    
    def _evaluate_state(self, state_tree: Tree,
                        state_history: list[State],
                        return_state_transformed=False) -> tuple[list[float], float]:
        '''evaluate a state
        
        return_state_transformed : flag that return the state transformed to a tensor'''

        # special evaluation if is final
        if self.game.is_terminal(state_tree.state):
            if state_tree.state[2] >= 0:
                state_value = 0
            else: # from the perspective of the state player. if it's final is always lose or draw
                state_value = -1

            if return_state_transformed:
                return [], state_value, True, state_tree.transform(state_history)
            else:
                return [], state_value, True

        state_transformed = state_tree.transform(state_history)
        state_value, policy = self.model.predict(state_transformed)
        
        policy = policy_matrix_to_policy(policy, state_tree.actions)
        policy = [float(p.numpy()) for p in policy]
        state_value = float(state_value)
        final = False

        # the Game class for tablut does not include history so the check about replicated states is here
        if (state_tree.state[0] != self.player and
            (any(self.state_equality(state_tree.state, sh)
                for sh in state_history[:-1]) or
            any(self.state_equality(state_tree.state, sh)
                for sh in state_tree.get_parents_state()[:-1]))):
            parent, action = state_tree.parent_action
            state_value = 1
            parent.P[action] = 0


        if return_state_transformed:
            return policy, state_value, final, state_transformed
        else:
            return policy, state_value, final

    def state_equality(self, s1: State, s2: State) -> bool:
        '''utility function for state equality. check only board'''
        for x in range(len(s1[1])):
            for y in range(len(s1[1][0])):
                if s1[1][x][y] != s2[1][x][y]:
                    return False
        return True