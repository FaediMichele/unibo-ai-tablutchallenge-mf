import random
import time
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tqdm import tqdm
import numpy as np
import json
import os
import pickle
import datetime
from games.tablut_simple.game import State, Action
import requests
import fcntl

from .util import policy_to_policy_matrix, actions_to_indexes, SAVE_ITERATIONS, BOARD_SIZE

class Model:
    # TODO do a reg net instead
    def __init__(self, path: str=None, remote=False, server_url="http://0.0.0.0:6000") -> None:
        if path is None:
            path = f'{"" if BOARD_SIZE == 9 else "simple_"}alpha_zero'
        self.model: keras.Model = None
        self.optimizer: keras.optimizers.Optimizer = None
        self.path = path
        self.remote = remote
        self.server_url = server_url
        self.first_game = datetime.datetime.now()
        if not remote:
            self.load_model()

    def load_model(self):
        if self.remote:
            return
        if self.optimizer is None:
            self.optimizer = keras.optimizers.Adam()
        current_model_path = self.get_last_model_path()
        if current_model_path is not None:
            model = keras.models.load_model(current_model_path, compile=False)
        else:
            raise Exception("Alpha Zero must be pre-trained. The neural network must be created somewhere else")
        self.model = model

    
    def save_model(self):
        # on remote the model file is managed by the server
        if self.remote:
            return
        
        config = self.load_config_file()
        current_model_path = os.path.join(self.path, config['current'])
        self.model.save(current_model_path, save_format='tf')
        if len(config['wins']) in SAVE_ITERATIONS:
            self.model.save(os.path.join(self.path,
                                         f'old_model_{len(config["wins"])}'),
                                         save_format='tf')
        if 'last_update' not in config or self.first_game is not None:
            last_update = self.first_game
            self.first_game = None
            
        else: 
            last_update = datetime.datetime.strptime(config['last_update'],
                                                    "%Y-%m-%d %H:%M:%S")
        config['total_seconds'] += (datetime.datetime.now() - last_update
                                    ).total_seconds()
        config['last_update'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_config_file(config)

    def load_config_file(self) -> dict:
        config_path = os.path.join(self.path, 'config.json')
        
        if os.path.isfile(config_path):
            return self.sync_file_op(config_path, 'r', lambda f: json.load(f))
        else:
            config = {
                    'current': 'last_model',
                    'wins': [],
                    'total_seconds': 0,
                    'last_update': datetime.datetime.now().strftime(
                                                    "%Y-%m-%d %H:%M:%S")
            }
            self.sync_file_op(config_path, 'w', lambda f: json.dump(config, f))
            return config
        
    def save_config_file(self, config: dict):
        config_path = os.path.join(self.path, 'config.json')
        self.sync_file_op(config_path, "w", lambda f: json.dump(config, f, indent=4))
    
    def player_wins(self, player: str, remaining_moves: int):
        config = self.load_config_file()
        config['wins'].append(f"{player}-{-remaining_moves}")
        self.save_config_file(config)

    def predict(self, data):
        data = tf.expand_dims(data, axis=0)
        if self.remote:
            value, policy = self.send_predict(data)
        else:
            value, policy = self.model.predict(data)
            value = value[0]
            policy = policy[0]
        policy = tf.nn.softmax(policy)
        if tf.reduce_any(tf.math.is_nan(policy)):
            print(data, "\n", policy)
        return value[0], policy
    
    def send_train_remote(self, data):
        post_response = requests.post(f"{self.server_url}/train_episode?model_path={self.path}",
                                      data=pickle.dumps(data))
        if post_response.status_code not in [200, 202]:
            raise Exception(post_response.text)

    def send_predict(self, data):
        post_data = {
            'model_path': self.path,
            'data': data.numpy().tolist()
        }
        headers = {'Content-Type': 'application/json'}
        post_response = requests.post(f"{self.server_url}/predict",
                                      data=json.dumps(post_data),
                                      headers=headers)
        if post_response.status_code != 200:
            raise Exception(post_response.text)
        state_value, policy = pickle.loads(post_response.content)

        return state_value, policy
    
    
    @tf.function
    def loss_zero(self, pi, z, v_pred, p_pred):        
        mse = tf.reduce_mean(keras.losses.mean_squared_error(z, v_pred))
        p_pred_prob = tf.nn.softmax(p_pred)
        cross_entropy = tf.reduce_mean(keras.losses.categorical_crossentropy(pi, p_pred_prob))
        penalty = sum(self.model.losses)

        return mse + cross_entropy + penalty
    
    def train_model(self,
                    batch_size: int=32,
                    step_for_epoch: int=100,
                    epochs: int=1):

        last_samples = self.get_last_samples(batch_size * step_for_epoch)
        
        for e in range(epochs):
            (states_tensor, policies_tensor,
             action_taken_tensor, zs_tensor)  = self.sample(last_samples, 0.25)
            
            with tqdm(total=states_tensor.shape[0] // batch_size,
                  desc="First episode") as pbar:
                
                for batch_start in range(policies_tensor.shape[0], step=batch_size):
                    batch_state = states_tensor[batch_start:batch_start + batch_size]
                    batch_policy = policies_tensor[batch_start:batch_start + batch_size]
                    batch_z = zs_tensor[batch_start:batch_start + batch_size]
                    batch_action_taken = action_taken_tensor[batch_start:batch_start + batch_size]

                    loss = self.train_batch(batch_state, batch_policy, batch_z, batch_action_taken)

                    pbar.set_description(f"Loss: {loss:.5e}")
                    pbar.update()


    def train_episode(self, states: list[tuple[tf.Tensor,
                                               list[float],
                                               list[Action],
                                               Action]],
                            batch_size: int=32):
        
        if self.remote:
            self.send_train_remote(states)
            return

        episode_states, episode_policy, episode_actions, episode_actions_taken, episode_z = zip(*states)

        state_matrix = tf.stack(episode_states, axis=0)
        policy_matrix = tf.stack([policy_to_policy_matrix(p, a)
                         for p, a in zip(episode_policy,
                                         episode_actions)], axis=0)
        
        action_takens_matrix = tf.stack(actions_to_indexes(episode_actions_taken), axis=0)
        z_matrix = tf.constant(episode_z, shape=(len(episode_z), 1))
        policy_matrix = tf.reshape(policy_matrix, (-1, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE * 2))


        
        (states_tensor, policies_tensor,
         action_taken_tensor, zs_tensor) = self.sample((state_matrix, policy_matrix,
                                                         action_takens_matrix, z_matrix),
                                                         0.25)
        
        with tqdm(total=states_tensor.shape[0] // batch_size,
                desc="First episode") as pbar:
            
            for batch_start in range(0, policies_tensor.shape[0], batch_size):
                batch_state = states_tensor[batch_start:batch_start + batch_size]
                batch_policy = policies_tensor[batch_start:batch_start + batch_size]
                batch_z = zs_tensor[batch_start:batch_start + batch_size]
                batch_action_taken = action_taken_tensor[batch_start:batch_start + batch_size]

                loss = self.train_batch(batch_state, batch_policy, batch_z, batch_action_taken)

                pbar.set_description(f"Loss: {loss:.5e}")
                pbar.update()

    def train_batch(self,
                    batch_state,
                    batch_policy,
                    batch_z,
                    batch_action_taken):
        
        with tf.GradientTape() as tape:
            v_pred, p_pred = self.model(batch_state, training=True)
            loss = self.loss_zero(batch_policy,
                                    batch_z,
                                    v_pred,
                                    p_pred)
            
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss


    def get_last_samples(self, number_of_samples: int=100_000
                         ) -> tuple[list[tf.Tensor],
                                    list[tf.Tensor],
                                    list[tf.Tensor]]:
        file_indexes = []
        for filename in os.listdir(self.path):
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
            'actions_taken': [],
            'z': [],
        }
        for filename, _ in sorted(file_indexes, key=lambda x: x[1],
                                  reverse=True):
            file_data = self.sync_file_op(os.path.join(self.path, filename), "rb",
                              lambda f: pickle.load(f))

            file_state, file_policy, file_actions, file_actions_taken, file_z = zip(*file_data)
            data_loaded['state'].extend(file_state)
            data_loaded['policy'].extend(file_policy)
            data_loaded['actions'].extend(file_actions)
            data_loaded['actions_taken'].extend(file_actions_taken)
            data_loaded['z'].extend(file_z)
            if len(data_loaded['z']) >= number_of_samples:
                
                data_loaded['state'] = data_loaded['state'][:number_of_samples]
                data_loaded['policy'] = data_loaded['policy'][:number_of_samples]
                data_loaded['actions'] = data_loaded['actions'][:number_of_samples]
                data_loaded['actions_taken'] = data_loaded['actions_taken'][:number_of_samples]
                data_loaded['z'] = data_loaded['z'][:number_of_samples]
                break
                         
        state_matrix = tf.stack(data_loaded['state'], axis=0)
        policy_matrix = tf.stack([policy_to_policy_matrix(p, a)
                         for p, a in zip(data_loaded['policy'],
                                         data_loaded['actions'])], axis=0)
        
        action_takens_matrix = tf.stack(actions_to_indexes(data_loaded['actions_taken']), axis=0)
        z_matrix = tf.constant(data_loaded['z'], shape=(len(data_loaded['z']), 1))
        policy_matrix = tf.reshape(policy_matrix, (-1, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE * 2))
        return state_matrix, policy_matrix, action_takens_matrix, z_matrix

    def sample(self, samples: tuple, final_size: float) -> tuple:
        states, policies, action_takens, zs = samples
        num_element = int(states.shape[0] * final_size)
        sum_prob = states.shape[0] * (states.shape[0] + 1) / 2
        choiches = np.random.choice(range(0, states.shape[0]), num_element,
                         p=[i/sum_prob for i in range(1, states.shape[0]+1)])
        mask = [i in choiches for i in range(0, states.shape[0])]
        return (tf.boolean_mask(states, mask),
                tf.boolean_mask(policies, mask),
                tf.boolean_mask(action_takens, mask),
                tf.boolean_mask(zs, mask))
    
    def get_last_model_path(self) -> str | None:
        config = self.load_config_file()
        current_model_path = os.path.join(self.path, config['current'])
        if os.path.isdir(current_model_path):
            return current_model_path
        return None
    
    def sync_file_op(self, file_path, mode, func,  retry_time=0.01):
        while True:
            try:
                with open(file_path, mode) as f:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    data = func(f)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    break
            except IOError:
                time.sleep(retry_time)
        return data

    def save_cache(self, cache: list[list], cache_mb_size:int=20):
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
        self.sync_file_op(cache_filename, "wb",
                          lambda f: pickle.dump(cache_data, f))