import time
from games.tictactoe.game import Action
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tqdm import tqdm
import numpy as np
import json
import os
import pickle
import datetime
import random
from more_itertools import ichunked
import requests
import fcntl
import shutil
from typing import Callable, Union, Any

from .util import boolean_mask_from_coordinates, no_repetition_actions, correct_actions, actions_to_indexes, PREVIOUS_STATE_TO_MODEL, SAVE_ITERATIONS, BOARD_SIZE
OLD_MODEL_SAVED = 10

class Model:
    '''Model container with utility function pertaining the model perdiction and training'''
    def __init__(self, path: str=None, old_model: bool=False, remote=False, server_url="http://0.0.0.0:6000", train_no_repetition=False) -> None:
        '''
        path : path that contains all the data for the model. such as keras model, model backup, cache, stats file
        old_model : flag to load an old version of the model
        remote : flag to send prediction and train episode to an http server
        server_url : url of the http server
        '''
        if path is None:
            path = f'models/{"" if BOARD_SIZE == 9 else "simple_"}reinforce_tictactoe'
        self.model: keras.Model = None
        self.optimizer: keras.optimizers.Optimizer = None
        self.path = path
        self.first_game = True
        self.remote = remote
        self.train_no_repetition = train_no_repetition
        self.server_url = server_url
        self.old_model = old_model
        if not remote:
            self.load_model(old_model)

    def load_model(self, old_model: bool):
        '''Load from memory a keras model'''
        if self.optimizer is None:
            self.optimizer = keras.optimizers.Adam(learning_rate=1e-4)

        model_path = self.get_last_model_path(old_model)
        if model_path is not None:
            model = keras.models.load_model(model_path, compile=False)
        else:
            model = self.create_model()
        self.model = model

    def save_model(self, path: str=None):
        '''Save model to memory - also save a backup of the model '''

        # on remote the model file is managed by the server
        if self.remote:
            return
        
        if path is None:
            path = self.path
        config = self.load_config_file()
        current_model_path = os.path.join(path, config['current'])


        for k in reversed(range(OLD_MODEL_SAVED)):
            old_model_path = os.path.join(path, f"old_model_-{k}")
            if os.path.isdir(old_model_path):
                if k == OLD_MODEL_SAVED - 1:
                    shutil.rmtree(old_model_path)
                else:
                    os.rename(old_model_path, os.path.join(path, f"old_model_-{k + 1}"))
        if os.path.isdir(current_model_path):
            os.rename(current_model_path, os.path.join(path, "old_model_-0"))

        self.model.save(current_model_path, save_format='tf')
        if len(config['wins']) in SAVE_ITERATIONS:
            self.model.save(os.path.join(path,
                                         f'old_model_{len(config["wins"])}'),
                                         save_format='tf')
        if 'last_update' not in config or self.first_game:
            self.first_game = False
            last_update = datetime.datetime.now()
        else: 
            last_update = datetime.datetime.strptime(config['last_update'],
                                                    "%Y-%m-%d %H:%M:%S")
        config['total_seconds'] += (datetime.datetime.now() - last_update
                                    ).total_seconds()
        config['last_update'] = datetime.datetime.now().strftime(
                                                        "%Y-%m-%d %H:%M:%S")
        self.save_config_file(config)

    def load_config_file(self) -> dict:
        '''Load or create a file containing the self play results'''
        config_path = os.path.join(self.path, 'config.json')
        
        if os.path.isfile(config_path):
            return self.sync_file_op(config_path, "r", lambda f: json.load(f))
        else:
            config = {
                    'current': 'last_model',
                    'wins': [],
                    'total_seconds': 0,
                    'last_update': datetime.datetime.now().strftime(
                                                    "%Y-%m-%d %H:%M:%S")
            }
            self.sync_file_op(config_path, "w", lambda f: json.dump(config, f, indent=4))
                
            return config
        
    def save_config_file(self, config: dict):
        '''Save the file containing the self play results'''
        config_path = os.path.join(self.path, 'config.json')
        self.sync_file_op(config_path, "w", lambda f: json.dump(config, f, indent=4))
    
    def player_wins(self, player: str, remaining_moves: int):
        '''Update the stats file with a new episode result'''
        config = self.load_config_file()
        config['wins'].append(f"{player}-{-remaining_moves}")
        self.save_config_file(config)


    def predict(self, data):
        '''Predict using the keras model or send the prediction to the http server'''
        data = tf.expand_dims(data, axis=0)
        if self.remote:
            value, policy = self.send_predict(data)
        else:
            value, policy = self.model.predict(data)
            value = value[0]
            policy = policy[0]

        policy = tf.reshape(policy, [BOARD_SIZE * BOARD_SIZE])
        policy = tf.nn.softmax(policy, axis=0)
        policy = tf.reshape(policy, [BOARD_SIZE, BOARD_SIZE])
        
        if tf.reduce_any(tf.math.is_nan(policy)):
            print(data, "\n", policy)
        return value[0], policy
    
    def send_train_remote(self, data):
        '''Send train message and data to the http server'''
        post_response = requests.post(f"{self.server_url}/train_episode?model_path={self.path}",
                                      data=pickle.dumps(data))
        if post_response.status_code not in [200, 202]:
            raise Exception(post_response.text)
        
    def send_predict(self, data):
        '''Send predict message and data to the http server'''
        post_data = {
            'model_path': self.path,
            'data': data.numpy().tolist(),
            'old_model': self.old_model
        }
        headers = {'Content-Type': 'application/json'}
        retry = 0
        while retry < 1000:
            try:
                post_response = requests.post(f"{self.server_url}/predict",
                                              data=json.dumps(post_data),
                                              headers=headers)
                if post_response.status_code != 200:
                    raise Exception(post_response.text)
                state_value, policy = pickle.loads(post_response.content)
                break
            except Exception as e:
                print(str(e))
                time.sleep(1)
                retry += 1

        return state_value, policy

    
    def loss_fn(self, states, z, v_pred, p_pred, actions_tensor):
        '''Loss of REINFORCE'''
        # The d value must not be optimized, is considered as a constant
        d = tf.stop_gradient(z - v_pred)
        state_value_loss = -tf.reduce_mean(v_pred * d)

        mse = tf.reduce_mean(tf.keras.metrics.mean_squared_error(z, v_pred))
        
        # shape = batch, height x width x (height + width (orthogonal slide))
        p_pred_flat = tf.reshape(p_pred, [p_pred.shape[0], BOARD_SIZE * BOARD_SIZE])
        p_pred_flat = tf.nn.log_softmax(p_pred_flat, axis=1)
        p_softmax = tf.reshape(p_pred_flat, [p_pred.shape[0], BOARD_SIZE, BOARD_SIZE])

        r = tf.expand_dims(tf.range(actions_tensor.shape[0]), axis=1)
        indices = tf.concat([r, actions_tensor], axis=1)
        selected_log_policy = tf.gather_nd(p_softmax, indices)
        
        policy_loss = -tf.reduce_mean(selected_log_policy * d)

        # Discourage action that leads to repeated states
        if self.train_no_repetition:
            action_repeated = [no_repetition_actions(states[idx]) for idx in range(z.shape[0])]
            mask = boolean_mask_from_coordinates(action_repeated)
            repeated_actions = tf.reduce_sum(tf.boolean_mask(p_softmax, mask))

            actions_correct = [correct_actions(states[idx]) for idx in range(z.shape[0])]
            mask_actions_correct = boolean_mask_from_coordinates(actions_correct, positives=False)
            not_actions_correct = tf.reduce_sum(tf.boolean_mask(p_softmax, mask_actions_correct))

            return state_value_loss + policy_loss + repeated_actions + not_actions_correct, policy_loss, state_value_loss, mse
        else:
            return state_value_loss + policy_loss, policy_loss, state_value_loss, mse
        
        
    
    
    def train_episode(self, state_action_z: list[tuple[tf.Tensor, Action, float]], batch_size:int=32) -> float:
        '''Train the model with the data of a single episode.
        DEPRECATED. it is better to play some episodes and then call train_model:
        it shuffle the data causing a faster convergence'''
        if self.remote:
            self.send_train_remote(state_action_z)
            return
        

        random.shuffle(state_action_z)

        with tqdm(total=len(state_action_z) // batch_size,
                  desc="First episode") as pbar:
            p_sum = 0
            v_sum = 0
            d_sum = 0
            for k, batch in enumerate(ichunked(state_action_z,
                                               batch_size)):
                states_batch, actions_batch, z_batch = zip(*batch)
                p_loss, v_loss, d_loss = self.train_batch(states_batch,
                                                          actions_batch,
                                                          z_batch)
                p_sum += float(p_loss.numpy())
                v_sum += float(v_loss.numpy())
                d_sum += float(d_loss.numpy())
                pbar.set_description(f"Value Loss: {v_sum / (k + 1):.5e}, "
                                     f"Policy loss: {p_sum / (k + 1):.5e}, "
                                     f"D value: {d_sum / (k + 1):.5e}")
                pbar.update()

    def train_model(self,
                    batch_size: int=32,
                    step_for_epoch: int=30,
                    epochs: int=1):
        '''Train the model with previouslt stored data'''

        last_samples = self.get_last_samples(step_for_epoch)
        data = []
        for states, z in last_samples:
            for s, a in states:
                data.append((s, a, z))
        random.shuffle(data)

        for e in range(epochs):
            with tqdm(total=len(data) // batch_size,
                      desc="First episode") as pbar:
                p_sum = 0
                v_sum = 0
                d_sum = 0
                for k, batch in enumerate(ichunked(data, batch_size)):
                    states_batch, actions_batch, zs_batch = zip(*batch)
                    p_loss, v_loss, d_loss = self.train_batch(states_batch,
                                                              actions_batch,
                                                              zs_batch)
                    p_sum += float(p_loss)
                    v_sum += float(v_loss)
                    d_sum += float(d_loss)
                    pbar.set_description(f"Value Loss: {v_sum / (k + 1):.5e}, "
                                        f"Policy loss: {p_sum / (k + 1):.5e}, "
                                        f"D value: {d_sum / (k + 1):.5e}")
                    pbar.update()
            print(f"Done epoch {e+1}/{epochs}")


    def train_batch(self, states: list[tf.Tensor], actions: list[Action],
                    z: list[float]):
        '''Compute the loss and update the weights with the data for a single batch.
        '''
        states_batch = tf.stack(states)
        z_batch = tf.constant(z, shape=(len(z), 1), dtype=tf.float32)
        actions_tensor = tf.constant(actions_to_indexes(actions))
        with tf.GradientTape() as tape:
            v_pred, p_pred = self.model(states_batch, training=True)
            loss, policy_loss, state_value_loss, d_loss = self.loss_fn(states_batch,
                                                            z_batch,
                                                            v_pred,
                                                            p_pred,
                                                            actions_tensor)
            
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return policy_loss, state_value_loss, d_loss

    def get_last_samples(self, number_of_games: int=1000,
                         max_episode_length: int=20
                         ) -> list[tuple[tf.Tensor, tf.Tensor], float]:
        ''' Load from disk the data of previos matches.
        
        number_of_samples : number of episode to load from memory
        max_episode_length : load episode the match no longer than this value (load some longer matches with score -0.1)'''
        file_indexes = []
        for filename in os.listdir(self.path):
            if filename.startswith("cache_") and filename.endswith(".pickle"):
                try:
                    n = int(filename[len("cache_"):-len(".pickle")])
                    file_indexes.append((filename, n))
                except ValueError:
                    pass  # Ignore non-integer filenames

        data_loaded = []
        for filename, _ in sorted(file_indexes, key=lambda x: x[1],
                                  reverse=True):
            
            file_games = self.sync_file_op(os.path.join(self.path, filename), "rb",
                                           lambda f: pickle.load(f))
            for state_action, z in file_games:
                if len(state_action) < max_episode_length:
                    data_loaded.append((state_action, z))
                elif len(state_action) < max_episode_length * 1.2:
                    data_loaded.append((state_action[:max_episode_length], -0.1))
            if len(data_loaded) >= number_of_games:
                data_loaded = data_loaded[:number_of_games]
                break
            print(filename)
        return data_loaded
    
    def get_last_model_path(self, old_model: bool) -> Union[str, None]:
        '''Get the path of the model to use.

        old_model : flag to load an old model
        '''
        config = self.load_config_file()

        if old_model:
            for k in reversed(range(OLD_MODEL_SAVED + 1)):
                old_model_path = os.path.join(self.path, f"old_model_-{k}")
                if os.path.isdir(old_model_path):
                    return old_model_path

        current_model_path = os.path.join(self.path, config['current'])
        if os.path.isdir(current_model_path):
            return current_model_path
        return None
        

    def sync_file_op(self, file_path: str,
                     mode: str,
                     func: Callable[[Any], Any], 
                     retry_time: float=0.01):
        '''Operation on a file in a sync way with concurrent processes.
        work only on LINUX (see fcntl)
        
        file_path : path to lock the file
        mode : flag for open function ('r', 'w', 'rb', 'wb', ...)
        retry_time : seconds to wait to retry the lock
        '''
        while True:
            try:
                with open(file_path, mode) as f:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    data = func(f)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    break
            except IOError:
                time.sleep(retry_time)
            except EOFError:
                time.sleep(retry_time)

        return data

    def save_cache(self, cache: list[tuple[tf.Tensor, float]],
                   cache_mb_size:int=20):
        '''Save data of a episode to memory
        
        cache_md_size : if a file is bigger than this value in mbyte the data is saved in another file
        '''
        # Made with ChatGPT ;)
        # Find the latest cache file in the folder
        latest_cache_file = None
        latest_cache_n = -1
        for filename in os.listdir(self.path):
            if filename.startswith("cache_") and filename.endswith(".pickle"):
                try:
                    n = int(filename[len("cache_"):-len(".pickle")])
                    if n > latest_cache_n:
                        latest_cache_n = n
                        latest_cache_file = os.path.join(self.path, filename)
                except ValueError:
                    pass  # Ignore non-integer filenames
        
        # If there's no cache file in the folder, start from scratch
        if latest_cache_file is None:
            cache_filename = os.path.join(self.path, "cache_0.pickle")
            cache_data = []
        else:
            # Check if the latest cache file is too big
            if os.path.getsize(latest_cache_file) > cache_mb_size * 1024 * 1024:
                # If the latest cache file is too big, create a new one
                cache_filename = os.path.join(self.path,
                                              f"cache_{latest_cache_n+1}.pickle")
                cache_data = []
            else:
                # Load the latest cache file
                cache_data = self.sync_file_op(latest_cache_file, "rb", lambda f: pickle.load(f))

                # Otherwise, append to the latest cache file
                cache_filename = latest_cache_file
        
        # Append the new data to the cache
        cache_data.extend(cache)
        self.sync_file_op(cache_filename, "wb", lambda f: pickle.dump(cache_data, f))


    def create_model(self) -> keras.Model:
        '''Create the keras model'''
        input_shape = (BOARD_SIZE, BOARD_SIZE, PREVIOUS_STATE_TO_MODEL + 2)
        policy_shape = (BOARD_SIZE, BOARD_SIZE)
        
        input_layer = keras.Input(shape=input_shape)
        
        policy_output = self.create_policy_network(input_shape, policy_shape)(input_layer)
        state_value_output = self.create_value_network(input_shape)(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=[state_value_output, policy_output])
        model.compile(loss=self.loss_fn)
        return model
        

    def create_policy_network(self, input_shape, policy_shape):
        '''The policy network is a reg net y like network'''
        # Made with ChatGPT with few corrections ;)
        inputs = tf.keras.Input(shape=input_shape)

        # Stem
        x = layers.Conv2D(filters=32, kernel_size=3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Main Blocks
        num_blocks = 3  # Number of blocks in the network
        num_filters = 32  # Initial number of filters

        for i in range(num_blocks):
            # Residual Path
            residual = x

            # Middle Block
            x = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

            x = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)

            # Skip Connection
            if i % 2 == 0 and i > 0:
                residual = layers.Conv2D(filters=num_filters, kernel_size=1, padding='same')(residual)
                residual = layers.BatchNormalization()(residual)

                x = layers.Add()([x, residual])
            x = layers.ReLU()(x)

            # Increase number of filters
            num_filters *= 2
        x = layers.GlobalAveragePooling2D()(x)
        dense_shape = policy_shape[0] * policy_shape[1]
        x = layers.Dense(dense_shape, activation='relu')(x)
        x = layers.Reshape(policy_shape)(x)
        

        return tf.keras.Model(inputs=inputs, outputs=x)

    def create_value_network(self, input_shape):
        '''The value network is a reg net y like network'''
        # Made with ChatGPT with few corrections ;)
        inputs = tf.keras.Input(shape=input_shape)

        # Stem
        x = layers.Conv2D(filters=32, kernel_size=3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Main Blocks
        num_blocks = 3  # Number of blocks in the network
        num_filters = 32  # Initial number of filters

        for i in range(num_blocks):
            # Residual Path
            residual = x

            # Middle Block
            x = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

            x = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)

            # Skip Connection
            if i % 2 == 0 and i > 0:
                residual = layers.Conv2D(filters=num_filters, kernel_size=1, padding='same')(residual)
                residual = layers.BatchNormalization()(residual)
                x = layers.Add()([x, residual])
            x = layers.ReLU()(x)

            # Increase number of filters
            num_filters *= 2
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(1, activation='linear')(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
