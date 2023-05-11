from games.board import Board
from games.game import Game
from games.player import Player, State, Action
import random
import math
from collections import defaultdict
import numpy as np
from more_itertools import ichunked
import os
import pickle
import logging
from collections.abc import Callable
from typing import Any
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tqdm import tqdm
import datetime


MAX_ACTION_COUNT = 10_000
PREVIOUS_STATE_TO_MODEL = 7
cache = []
N: dict[tuple[State, Action], int] = defaultdict(lambda: 0)
W: dict[tuple[State, Action], float] = defaultdict(lambda: 0.0)
Q: dict[tuple[State, Action], float] = defaultdict(lambda: 0.0)
P: dict[tuple[State, Action], float] = dict()
parent_child: dict[tuple[State, Action], State] = dict()
child_parent: dict[State, tuple[State, Action]] = dict()
state_actions: dict[State, list[Action]] = dict()

class AlphaTablutZero(Player):
    ''' Player that take random actions '''

    def __init__(self, make_move: Callable[[None | State |Action], 
                                           None],
                                           board: Board,
                                           game: Game,
                                           player: str,
                                           train: bool=False,
                                           ms_for_search: int=5000):
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
        self.cache: list[list[State, list[float], None | float]] = []
        
        self.model = ModelUtil.load_model()
        self.train = train
        super(Player, self).__init__()

    def next_action(self, last_action: Action, state_history: list[State]):
        ''' Function called when the opponent take the move and now is the
        turn of this player

        Keyword arguments:
        last_action -- the last move that the opponent have take
        '''
        state = state_hashable(self.board.state)
        self._remove_old_branch(state)
        policy = self._mcts(state, state_history, self.ms_for_search)
        if self.train:
            self.make_move(random.choices(state_actions[state], policy)[0])
        else:
            self.make_move(state_actions[state][argmax(policy)])

    def end(self, last_action: Action, winning: str):
        """Called when a player wins.

        last_action -- the winning move
        winning -- the winning player
        """
        logging.info(f'Calling win on {Player}, winning: {winning}')
        for evaluation in self.cache:
            evaluation.append(1.0 if winning == self.player else -1.0)

        if self.train:
            ModelUtil.train_model(self.model, self.cache)

    

    def _remove_old_branch(self, current_state: State):
        global N, W, Q, P, parent_child, child_parent, state_actions
        visited_states_for_branch = self._descendant(current_state)

        tmp = defaultdict(lambda: 0.0)
        for (s,a), v in N.items():
            if s in visited_states_for_branch:
                tmp[(s, a)] = v
        N = tmp

        tmp = defaultdict(lambda: 0.0)
        for (s,a), v in W.items():
            if s in visited_states_for_branch:
                tmp[(s, a)] = v
        W = tmp

        tmp = defaultdict(lambda: 0.0)
        for (s,a), v in Q.items():
            if s in visited_states_for_branch:
                tmp[(s, a)] = v
        Q = tmp

        P = dict(((s, a), v) for (s,a), v in P.items()
                 if s in visited_states_for_branch)
        
        parent_child = dict(
            ((s, a), v) for (s, a), v in parent_child.items()
            if s in visited_states_for_branch)

        child_parent = dict((s,v) for s,v in child_parent.items()
                                 if s in visited_states_for_branch)
        if current_state in child_parent.keys():
            del child_parent[current_state]
        state_actions = dict((s,v) for s,v in state_actions.items()
                                 if s in visited_states_for_branch)

    def _descendant(self, state) -> list[State]:
        if state in state_actions.keys():
            return list(self._descendant(child)
                        
                        for a in state_actions[state]
                        for child in parent_child[(state, a)]
                    ) + [state]
        else:
            return [state]
    def _mcts(self, root: State, state_history: list[State],
              ms_for_search: int, temperature: float=1.0) ->list[float]:
        global N, W, Q, P, parent_child, child_parent, state_actions
        state_actions[root] = self.game.actions(root)

        p_0, v_0 = self._evaluate_state(root, state_actions[root],
                                        state_history)

        self._expand(root, state_actions[root], p_0)
        
        start_timer = datetime.datetime.now()
        while (datetime.datetime.now() - start_timer
               ).total_seconds() * 1000 < ms_for_search:
            s = root
            tree_path = []
            while s in state_actions.keys():
                
                a_star = state_actions[s][
                    argmax([
                        Q[(s, a)] +
                        self._upper_confidence_bound(s, a)
                        for a in state_actions[s]])]
                N[(s, a_star)] = N[(s, a_star)] + 1
                s = parent_child[(s, a_star)]
                tree_path.append(s)

            state_actions[s] = self.game.actions(s)
            p, v = self._evaluate_state(s, state_actions[s],
                                        state_history + tree_path)
            self._expand(s, state_actions[s], p)
            self._backup(s, v)

        count_action_taken = [N[(root, a)] ** (1/temperature)
                              for a in state_actions[root]]
        denominator_policy = sum(count_action_taken)
        policy = [t / denominator_policy for t in count_action_taken]
        self.cache.append([
            root,
            policy
        ])
        return policy


    def _upper_confidence_bound(self, state: State, action: Action,
                                cput: float = 5.0) -> float:
        global N, P, state_actions
        return cput * P[(state, action)] * \
            math.sqrt(sum([N[(state, a)] for a in state_actions[state]])) \
                / (1 + N[(state, action)])

    def _expand(self, state: State, actions: list[Action], policy: list[float]):
        global P, parent_child, child_parent
        for a, pa in zip(actions, policy):
            new_state = state_hashable(self.game.result(unhash_state(state), a))
            parent_child[(state, a)] = new_state
            child_parent[new_state] = (state, a)
            P[(state, a)] = pa

    def _backup(self, state: State, state_value: float):
        while state in child_parent.keys():
            state, action = child_parent[state]
            W[(state, action)] = W[(state, action)] + state_value
            Q[(state, action)] = W[(state, action)] / N[(state, action)]

    def _evaluate_state(self, state: State, actions: list[Action],
                        state_history: list[State]
                        ) -> tuple[list[float], float]:
        state_value, policy = ModelUtil.predict(self.model,
                                                self._state_transoform(state,
                                                state_history))
        policy = self._policy_transform(policy, actions)
        return policy, state_value
    
    def _state_transoform(self, state: State, state_history: list[State]
                          ) -> tf.Tensor:
        player, board = state
        boards = [board] + [sh[1] for sh in state_history[:PREVIOUS_STATE_TO_MODEL]]
        while len(boards) <= PREVIOUS_STATE_TO_MODEL:
            boards.append(boards[-1])
        board_shape = (1, len(state[1]), len(state[1][0]))
        
        tensor_board = tf.constant(boards)
        tensor_player = tf.ones(board_shape, tensor_board.dtype) * (player * 2 - 1)
        return tf.expand_dims(tf.concat([tensor_board, tensor_player],
                                        axis=0), axis=0)


    def _policy_transform(self, policy, actions: list[Action]) -> list[float]:
        res = []
        for a in actions:
            res.append(policy[0, a[0], a[1], 0] * policy[0, a[2], a[3], 1])

        # normalize. Maybe a softmax is better?
        s_res = sum(res)
        res = [r/s_res for r in res]

        return res
    


class ModelUtil:
    # TODO do a reg net instead
    @staticmethod
    def create_model(board_size: tuple[int, int]
                     ) -> tuple[keras.Model, keras.optimizers.Optimizer]:
        # 1 channel for the current player and 1 for the current state
        input_shape = (PREVIOUS_STATE_TO_MODEL + 2, board_size[0], board_size[1])
        input_layer = keras.Input(input_shape) 
        x = layers.Conv2D(64, 3, 1, 'same', activation='swish')(input_layer)
        x = layers.Conv2D(64, 3, 1, 'same', activation='swish')(x)
        x = layers.Conv2D(64, 3, 1, 'same', activation='swish',
                          kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.Conv2D(64, 3, 1, 'same', activation='swish',
                          kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.Conv2D(64, 3, 1, 'same', activation='swish',
                          kernel_regularizer=keras.regularizers.l2(0.01))(x)
        
        value = layers.Dense(1, 'linear')(layers.Flatten()(x))
        policy = layers.Conv2D(2, 3, 1, 'same', activation='softmax')(x)
        model = keras.Model(inputs=input_layer, outputs=[value, policy])
        return model

    @staticmethod
    def load_model(path: str='alpha_zero', board_size: tuple[int, int]=(9, 9)
                   ) -> tuple[keras.Model, keras.optimizers.Optimizer]:
        last_model = ModelUtil.get_last_model_path(path)
        if last_model[0] is not None:
            model = keras.models.load_model(path, compile=False)
            optimizer = keras.optimizers.Adam()
            return model, optimizer
        else:
            model = ModelUtil.create_model(board_size)
            optimizer = keras.optimizers.Adam()
            return model, optimizer

    
    @staticmethod
    def save_model(model: tuple[keras.Model, keras.optimizers.Optimizer],
                   path: str='alpha_zero',
                   key: int=0):
        model_dir = ModelUtil.get_last_model_path(path)
        if model_dir is not None:
            path = model_dir[0].replace(model_dir[1], model_dir[1] + 1)
            model[0].save(path)
        else:
            path = os.path.join(path, f'model_{key}')
            model[0].save(path)
        


    @staticmethod
    def predict(model: tuple[keras.Model, keras.optimizers.Optimizer], data):
        value, policy = model[0].predict(data, verbose=0)
        return value, policy

    @staticmethod
    def train_model(model: tuple[keras.Model, keras.optimizers.Optimizer],
                    cache_folder: str='alpha_zero',
                    batch_size: int=32, step_for_epoch: int=1000,
                    epochs: int=1, cache: list[dict]=None):
        
        model, optimizer = model
        if cache is not None:
            ModelUtil.save_cache(cache, cache_folder)

        samples = ModelUtil.get_last_samples(cache_folder, batch_size *
                                             step_for_epoch * epochs)
        
        def loss_fn(pi, z, v, p):
            return (keras.losses.mean_squared_error(z, v) +
                    keras.losses.categorical_crossentropy(pi, p) +
                    sum(model.losses)) # penalty

        for e in range(epochs):
            for batch in tqdm(ichunked(samples, batch_size),
                              total=step_for_epoch):
                states_batch, pi_batch, z_batch = zip(*batch)

                with tf.GradientTape() as tape:
                    v_pred, p_pred = model(states_batch, training=True)
                    loss = loss_fn(pi_batch, z_batch, v_pred, p_pred)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            print(f"Done epoch {e+1}")

    @staticmethod
    def get_last_samples(folder_path: str, number_of_samples: int=100_000
                         ) -> list[list[State, list[float], float]]:
        file_indexes = []
        for filename in os.listdir(folder_path):
            if filename.startswith("cache_") and filename.endswith(".pickle"):
                try:
                    n = int(filename[len("cache_"):-len(".pickle")])
                    file_indexes.append((filename, n))
                except ValueError:
                    pass  # Ignore non-integer filenames

        samples = []
        for filename, _ in sorted(file_indexes, key=lambda x: x[1],
                                  reverse=True):
            with open(os.path.join(folder_path, filename), "rb") as f:
                samples.extend(pickle.load(f))

            if len(samples) >= number_of_samples:
                return samples[:number_of_samples]
            
        return samples
    
    @staticmethod
    def get_last_model_path(folder_path: str='alpha_zero') -> str | None:
        res_dir = None
        dir_n = -1
        for name in os.listdir(folder_path):
            if os.path.isdir(name) and name.startswith("model_"):
                try:
                    n = int(name[len("model_"):])
                    if n > dir_n:
                        dir_n = n
                        res_dir = os.path.join(folder_path, name)
                except ValueError:
                    pass  # Ignore non-integer filenames
        return res_dir, dir_n if dir_n > 0 else None

    @staticmethod
    def save_cache(cache: list[dict], folder_path: str):
        # Made with ChatGPT ;)
        # Find the latest cache file in the folder
        latest_cache_file = None
        latest_cache_n = -1
        for filename in os.listdir(folder_path):
            if filename.startswith("cache_") and filename.endswith(".pickle"):
                try:
                    n = int(filename[len("cache_"):-len(".pickle")])
                    if n > latest_cache_n:
                        latest_cache_n = n
                        latest_cache_file = os.path.join(folder_path, filename)
                except ValueError:
                    pass  # Ignore non-integer filenames
        
        # If there's no cache file in the folder, start from scratch
        if latest_cache_file is None:
            cache_filename = os.path.join(folder_path, "cache_0.pickle")
            cache_data = []
        else:
            # Check if the latest cache file is too big
            if os.path.getsize(latest_cache_file) > 20 * 1024 * 1024:
                # If the latest cache file is too big, create a new one
                cache_filename = os.path.join(folder_path,
                                              f"cache_{latest_cache_n+1}.pickle")
                cache_data = []
            else:
                # Load the latest cache file
                with open(latest_cache_file, "rb") as f:
                    cache_data = pickle.load(f)

                # Otherwise, append to the latest cache file
                cache_filename = latest_cache_file
        
        # Append the new data to the cache
        cache_data.extend(cache)
        with open(cache_filename, "wb") as f:
            pickle.dump(cache_data, f)

        


def state_hashable(state: State) -> tuple:
    player, board = state
    return player, tuple(tuple(line) for line in board)

def unhash_state(state) -> list:
    player, board = state
    return player, list(list(line) for line in board)


def argmax(lis: list[float]) -> int:
    k = 0
    for i in range(1, len(lis)):
        if lis[i] > lis[k]:
            k = i
    return k