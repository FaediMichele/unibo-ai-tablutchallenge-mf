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
import json

SAVE_ITERATIONS = [1, 10, 25, 50, 100, 300, 500, 1000, 2000, 5000, 10000]

MAX_ACTION_COUNT = 10_000
PREVIOUS_STATE_TO_MODEL = 7


class TransferableMemory:
    def __init__(self) -> None:
        self.cache = []
        self.N: dict[tuple[State, Action], int] = defaultdict(lambda: 0)
        self.W: dict[tuple[State, Action], float] = defaultdict(lambda: 0.0)
        self.Q: dict[tuple[State, Action], float] = defaultdict(lambda: 0.0)
        self.P: dict[tuple[State, Action], float] = dict()
        self.parent_child: dict[tuple[State, Action], State] = dict()
        self.child_parent: dict[State, tuple[State, Action]] = dict()
        self.state_actions: dict[State, list[Action]] = dict()

class AlphaTablutZero(Player):
    ''' Player that take random actions '''

    def __init__(self, make_move: Callable[[None | State |Action], 
                                           None],
                                           board: Board,
                                           game: Game,
                                           player: str,
                                           greedy: bool=False,
                                           memory: TransferableMemory=None,
                                           ms_for_search: int=5000,
                                           ):
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

    def _remove_old_branch(self, current_state: State):
        
        visited_states_for_branch = self._descendant(current_state)

        tmp = defaultdict(lambda: 0.0)
        for (s,a), v in self.memory.N.items():
            if s in visited_states_for_branch:
                tmp[(s, a)] = v
        self.memory.N = tmp

        tmp = defaultdict(lambda: 0.0)
        for (s,a), v in self.memory.W.items():
            if s in visited_states_for_branch:
                tmp[(s, a)] = v
        self.memory.W = tmp

        tmp = defaultdict(lambda: 0.0)
        for (s,a), v in self.memory.Q.items():
            if s in visited_states_for_branch:
                tmp[(s, a)] = v
        self.memory.Q = tmp

        self.memory.P = dict(((s, a), v) for (s,a), v in self.memory.P.items()
                 if s in visited_states_for_branch)
        
        self.memory.parent_child = dict(
            ((s, a), v) for (s, a), v in self.memory.parent_child.items()
            if s in visited_states_for_branch)

        self.memory.child_parent = dict((s,v) for s,v in
                                        self.memory.child_parent.items()
                                        if s in visited_states_for_branch)
        if current_state in self.memory.child_parent.keys():
            del self.memory.child_parent[current_state]
        self.memory.state_actions = dict((s,v) for s,v in
                                         self.memory.state_actions.items()
                                         if s in visited_states_for_branch)

    def _descendant(self, state) -> list[State]:
        if state in self.memory.state_actions.keys():
            return list(self._descendant(child)
                        for a in self.memory.state_actions[state]
                        for child in self.memory.parent_child[(state, a)]
                        ) + [state]
        else:
            return [state]

    def _mcts(self, root: State, state_history: list[State],
              ms_for_search: int, temperature: float=1.0) ->list[float]:
        self.memory.state_actions[root] = self.game.actions(root)

        p_0, v_0, root_tensor = self._evaluate_state(
            root, self.memory.state_actions[root], state_history,
            return_state_transformed=True)

        self._expand(root, self.memory.state_actions[root], p_0)
        
        start_timer = datetime.datetime.now()
        while (datetime.datetime.now() - start_timer
               ).total_seconds() * 1000 < ms_for_search:
            s = root
            tree_path = []
            while s in self.memory.state_actions.keys():
                
                a_star = argmax(self.memory.state_actions[s],
                                key=lambda a:self.memory. Q[(s, a)] + 
                                self._upper_confidence_bound(s, a))
                self.memory.N[(s, a_star)] = self.memory.N[(s, a_star)] + 1
                s = self.memory.parent_child[(s, a_star)]
                tree_path.append(s)

            self.memory.state_actions[s] = self.game.actions(s)
            p, v = self._evaluate_state(s, self.memory.state_actions[s],
                                        state_history + tree_path)
            self._expand(s, self.memory.state_actions[s], p)
            self._backup(s, v)

        count_action_taken = [self.memory.N[(root, a)] ** (1/temperature)
                              for a in self.memory.state_actions[root]]
        denominator_policy = sum(count_action_taken)
        policy = [t / denominator_policy for t in count_action_taken]
        self.cache.append([
            root_tensor,
            policy,
            self.memory.state_actions[root]
        ])
        return policy


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
            self.memory.child_parent[new_state] = (state, a)
            self.memory.P[(state, a)] = pa

    def _backup(self, state: State, state_value: float):
        while state in self.memory.child_parent.keys():
            state, action = self.memory.child_parent[state]
            self.memory.W[(state, action)] = self.memory.W[(state, action)] +\
                state_value
            self.memory.Q[(state, action)] = self.memory.W[(state, action)] /\
                self.memory.N[(state, action)]

    def _evaluate_state(self, state: State, actions: list[Action],
                        state_history: list[State],
                        return_state_transformed=False) -> tuple[list[float], float]:
        
        state_transformed = self._state_transoform(state, state_history)
        state_value, policy = ModelUtil.predict(model,
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
        board_shape = (1, len(state[1]), len(state[1][0]))
        
        tensor_board = tf.constant(boards)
        tensor_player = tf.ones(board_shape, tensor_board.dtype) * (player * 2 - 1)
        return tf.expand_dims(tf.concat([tensor_board, tensor_player],
                                        axis=0), axis=0)
    


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
        x = layers.Conv2D(81, 3, 1, 'same', activation='swish',
                          kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x_p = layers.Conv3D(9, 3, 1, 'same', activation='swish',
                          kernel_regularizer=keras.regularizers.l2(0.01))(tf.reshape(x, (-1,9,9,9,9)))
        x_p = layers.Conv3D(9, 3, 1, 'same', activation='swish',
                          kernel_regularizer=keras.regularizers.l2(0.01))(x_p)
        
        value = layers.Dense(1, 'linear')(layers.Flatten()(x))
        policy = layers.Conv3D(9, 3, 1, 'same', activation='softmax')(x_p)
        model = keras.Model(inputs=input_layer, outputs=[value, policy])
        model.compile()
        return model

    @staticmethod
    def load_model(path: str='alpha_zero', board_size: tuple[int, int]=(9, 9)
                   ) -> tuple[keras.Model, keras.optimizers.Optimizer]:
        current_model_path = ModelUtil.get_last_model_path(path)
        if current_model_path is not None:
            model = keras.models.load_model(current_model_path, compile=False)
            optimizer = keras.optimizers.Adam()
            return model, optimizer
        else:
            model = ModelUtil.create_model(board_size)
            optimizer = keras.optimizers.Adam()
            return model, optimizer

    
    @staticmethod
    def save_model(model: tuple[keras.Model, keras.optimizers.Optimizer],
                   path: str='alpha_zero'):
        config_file_path = os.path.join(path, 'config.json')
        if os.path.isfile(config_file_path):
            with open(os.path.join(path, 'config.json'), 'r') as f:
                config = json.load(f)
        else:
            config = {
                'current_iteration': 0,
                'current': 'last_model'
            }
        current_iteration = config['current_iteration']
        current_model_path = config['current']
        model[0].save(os.path.join(path, current_model_path))

        if current_iteration in SAVE_ITERATIONS:
            model[0].save(os.path.join(path, f'old_model_{current_iteration}'))

        config['current_iteration'] += 1
        with open(config_file_path, 'w') as f:
            json.dump(config, f)

    @staticmethod
    def predict(model: tuple[keras.Model, keras.optimizers.Optimizer], data):
        value, policy = model[0].predict(data, verbose=0)
        return value, policy

    @staticmethod
    @tf.function
    def loss_fn(model, pi, z, v, p):
        mse = keras.losses.mean_squared_error(z, v)
        pi, p = tf.reshape(pi, (pi.shape[0],-1)), tf.reshape(p, (p.shape[0],-1))
        cross_entropy = keras.losses.categorical_crossentropy(pi, p)
        penalty = sum(model.losses)
        return mse + cross_entropy + penalty

    @staticmethod
    def train_model(model: tuple[keras.Model, keras.optimizers.Optimizer],
                    cache: list[dict]=None,
                    cache_folder: str='alpha_zero',
                    batch_size: int=32,
                    step_for_epoch: int=100,
                    epochs: int=1):
        
        model, optimizer = model
        if cache is not None:
            ModelUtil.save_cache(cache, cache_folder)

        states_tensor, policies_tensor, zs_tensor  = ModelUtil.sample(
            ModelUtil.get_last_samples(cache_folder,
                                       batch_size * step_for_epoch), 0.3
        )

        print(policies_tensor.shape[0] // batch_size)
        for e in range(epochs):
            for batch_start in tqdm(range(0, policies_tensor.shape[0],
                                          batch_size)):
                batch_state = states_tensor[batch_start:batch_start +\
                                                   batch_size]
                batch_policy = policies_tensor[batch_start:batch_start + batch_size]
                batch_z = zs_tensor[batch_start:batch_start + batch_size]
                with tf.GradientTape() as tape:
                    v_pred, p_pred = model(batch_state, training=True)
                    loss = ModelUtil.loss_fn(model, batch_policy, batch_z, v_pred, p_pred)
                
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            print(f"Done epoch {e+1}")

    @staticmethod
    def get_last_samples(folder_path: str, number_of_samples: int=100_000
                         ) -> tuple[list[tf.Tensor], list[tf.Tensor], list[tf.Tensor]]:
        file_indexes = []
        for filename in os.listdir(folder_path):
            if filename.startswith("cache_") and filename.endswith(".pickle"):
                try:
                    n = int(filename[len("cache_"):-len(".pickle")])
                    file_indexes.append((filename, n))
                except ValueError:
                    pass  # Ignore non-integer filenames

        data_loaded = {
            'state': [],
            'policy': [],
            'actions': [],
            'z': []
        }
        for filename, _ in sorted(file_indexes, key=lambda x: x[1],
                                  reverse=True):
            with open(os.path.join(folder_path, filename), "rb") as f:
                file_data = pickle.load(f)

            file_state, file_policy, file_actions, file_z = zip(*file_data)
            data_loaded['state'].extend(file_state)
            data_loaded['policy'].extend(file_policy)
            data_loaded['actions'].extend(file_actions)
            data_loaded['z'].extend(file_z)
            print(len(data_loaded['z']), number_of_samples)
            if len(data_loaded['z']) >= number_of_samples:
                
                data_loaded['state'] = data_loaded['state'][:number_of_samples]
                data_loaded['policy'] = data_loaded['policy'][:number_of_samples]
                data_loaded['actions'] = data_loaded['actions'][:number_of_samples]
                data_loaded['z'] = data_loaded['z'][:number_of_samples]
                print(len(data_loaded['z']))
                break
        
        state_matrix = tf.concat(data_loaded['state'], axis=0)
        policy_matrix = tf.stack([policy_to_policy_matrix(p, a)
                         for p, a in zip(data_loaded['policy'],
                                         data_loaded['actions'])], axis=0)
        z_matrix = tf.constant(data_loaded['z'], shape=(len(data_loaded['z']), 1))
        return state_matrix, policy_matrix, z_matrix
    
    @staticmethod
    def sample(samples: tuple, final_size: float) -> tuple:
        states, policies, zs = samples
        num_element = int(states.shape[0] * final_size)
        sum_prob = states.shape[0] * (states.shape[0] + 1) / 2
        choiches = np.random.choice(range(0, states.shape[0]), num_element,
                         p=[i/sum_prob for i in range(1, states.shape[0]+1)])
        mask = [i in choiches for i in range(0, states.shape[0])]
        return (tf.boolean_mask(states, mask),
                tf.boolean_mask(policies, mask),
                tf.boolean_mask(zs, mask))
    
    @staticmethod
    def get_last_model_path(folder_path: str='alpha_zero') -> str | None:
        if os.path.isfile(os.path.join(folder_path, 'config.json')):
            with open(os.path.join(folder_path, 'config.json'), 'r') as f:
                config = json.load(f)
            if os.path.isdir(os.path.join(folder_path, config['current'])):
                return os.path.join(folder_path, config['current'])
        return None

    @staticmethod
    def save_cache(cache: list[list], folder_path: str, cache_mb_size:int=20):
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
            if os.path.getsize(latest_cache_file) > cache_mb_size * 1024 * 1024:
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

model = ModelUtil.load_model()


def state_hashable(state: State) -> tuple:
    player, board = state
    return player, tuple(tuple(line) for line in board)

def unhash_state(state) -> list:
    player, board = state
    return player, list(list(line) for line in board)


def argmax(elements: list[Any], key: Callable[[Any], float]=lambda x:x) -> int:
    k = 0
    k_val = key(elements[k])
    for i in range(1, len(elements)):
        i_val = key(elements[i])
        if i_val > k_val:
            k = i
            k_val = i_val
    return elements[k]

def policy_matrix_to_policy(matrix, actions: list[Action]) -> list[float]:
    return [matrix[a[0],a[1],a[2],a[3]] for a in actions]

def policy_to_policy_matrix(policy: list[float], actions: list[Action]
                            ) -> tf.Tensor:
    action_to_policy = dict((a, p) for a, p in zip(actions, policy))
    matrix = tf.constant([
        [
            [
                [action_to_policy[(x1,y1,x2,y2)]
                 if (x1,y1,x2,y2) in action_to_policy.keys()
                 else 0.0
                 for y2 in range(9)
                ] for x2 in range(9)
            ] for y1 in range(9)
        ] for x1 in range(9)
    ])
    return matrix    