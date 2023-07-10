import random
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tqdm import tqdm
import numpy as np
import json
import os
import pickle
import datetime
from games.tablut.game import State, Action
import requests
import fcntl
import shutil
from typing import Callable, Union, Any

from .util import policy_to_policy_matrix, actions_to_indexes, SAVE_ITERATIONS, BOARD_SIZE
OLD_MODEL_SAVED = 10

class Model:
    '''Model container with utility function pertaining the model perdiction and training'''

    def __init__(self, path: str=None, old_model: bool=False, remote=False, server_url="http://0.0.0.0:6000") -> None:
        '''
        path : path that contains all the data for the model. such as keras model, model backup, cache, stats file
        old_model : flag to load an old version of the model
        remote : flag to send prediction and train episode to an http server
        server_url : url of the http server
        '''
        if path is None:
            path = f'models/{"" if BOARD_SIZE == 9 else "simple_"}alpha_zero'
        self.model: keras.Model = None
        self.optimizer: keras.optimizers.Optimizer = None
        self.path = path
        self.remote = remote
        self.server_url = server_url
        self.first_game = datetime.datetime.now()
        self.old_model = old_model
        if not remote:
            self.load_model(old_model)

    def load_model(self, old_model: bool):
        '''Load from memory a keras model'''
        if self.remote:
            return
        if self.optimizer is None:
            self.optimizer = keras.optimizers.Adam(5e-4)
        current_model_path = self.get_last_model_path(old_model)
        if current_model_path is not None:
            model = keras.models.load_model(current_model_path, compile=False)
        else:
            raise Exception("Alpha Zero must be pre-trained. The neural network must be created somewhere else")
        self.model = model

    
    def save_model(self):
        '''Save model to memory - also save a backup of the model '''
        # on remote the model file is managed by the server
        if self.remote:
            return
        
        config = self.load_config_file()
        current_model_path = os.path.join(self.path, config['current'])


        for k in reversed(range(OLD_MODEL_SAVED)):
            old_model_path = os.path.join(self.path, f"old_model_-{k}")
            if os.path.isdir(old_model_path):
                if k == OLD_MODEL_SAVED - 1:
                    shutil.rmtree(old_model_path)
                else:
                    os.rename(old_model_path, os.path.join(self.path, f"old_model_-{k + 1}"))
        os.rename(current_model_path, os.path.join(self.path, "old_model_-0"))

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
        '''Load or create a file containing the self play results'''
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

        policy = tf.reshape(policy, [BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * 2])
        policy = tf.nn.softmax(policy, axis=0)
        # print(policy, value)
        policy = tf.reshape(policy, [BOARD_SIZE, BOARD_SIZE, BOARD_SIZE * 2])

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
    
    @tf.function
    def loss_fn(self, pi, z, v_pred, p_pred):        
        ''' Loss of Alpha Zero'''

        mse = tf.reduce_mean(keras.losses.mean_squared_error(z, v_pred))
        p_pred_flat = K.batch_flatten(p_pred)
        pi_flat = K.batch_flatten(pi)

        # p_pred_flat = tf.reshape(p_pred, [p_pred.shape[0], BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * 2])
        # pi_flat = tf.reshape(pi, [p_pred.shape[0], BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * 2])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pi_flat, logits=p_pred_flat))
        penalty = sum(self.model.losses)


        return mse + cross_entropy + penalty, mse, cross_entropy
    

    
    def train_model(self,
                    batch_size: int=32,
                    step_for_epoch: int=100,
                    epochs: int=1):
        '''Train the model with previouslt stored data'''

        last_samples = self.get_last_samples(batch_size * step_for_epoch)
        # state_matrix, policy_matrix, action_takens_matrix, z_matrix


        train_data, valid_data = self.train_test_split(last_samples, 0.85)

        for e in range(epochs):            
            
            (states_tensor, policies_tensor,
             action_taken_tensor, zs_tensor) = self.shuffle(train_data)
            (valid_states_tensor, valid_policies_tensor,
             valid_action_taken_tensor, valid_zs_tensor) = self.shuffle(valid_data)
            
            with tqdm(total=states_tensor.shape[0] // batch_size,
                  desc="First episode") as pbar:
                p_sum = 0
                v_sum = 0
                
                for k, batch_start in enumerate(range(0, policies_tensor.shape[0], batch_size)):
                    batch_state = states_tensor[batch_start:batch_start + batch_size]
                    batch_policy = policies_tensor[batch_start:batch_start + batch_size]
                    batch_z = zs_tensor[batch_start:batch_start + batch_size]
                    batch_action_taken = action_taken_tensor[batch_start:batch_start + batch_size]

                    loss, v_loss, p_loss = self.train_batch(batch_state, batch_policy, batch_z, batch_action_taken)

                    p_sum += p_loss
                    v_sum += v_loss
                    pbar.set_description(f"Value Loss: {v_sum / (k + 1):.5e}, "
                                        f"Policy loss: {p_sum / (k + 1):.5e}")
                    pbar.update()

            loss_sum = 0
            p_sum = 0
            v_sum = 0
            for k, batch_start in enumerate(range(0, valid_policies_tensor.shape[0], batch_size)):
                batch_state = valid_states_tensor[batch_start:batch_start + batch_size]
                batch_policy = valid_policies_tensor[batch_start:batch_start + batch_size]
                batch_z = valid_zs_tensor[batch_start:batch_start + batch_size]
                batch_action_taken = valid_action_taken_tensor[batch_start:batch_start + batch_size]

                v_pred, p_pred = self.model(batch_state, training=True)
                loss , v_loss, p_loss = self.loss_fn(batch_policy,
                                        batch_z,
                                        v_pred,
                                        p_pred)
                loss_sum += loss
                p_sum += p_loss
                v_sum += v_loss
            print(f"valid loss: {loss_sum / (k + 1):.5e}, "
                  f"valid value Loss: {v_sum / (k + 1):.5e}, "
                  f"valid policy loss: {p_sum / (k + 1):.5e}")
                


    def train_episode(self, states: list[tuple[tf.Tensor,
                                               list[float],
                                               list[Action],
                                               Action]],
                            batch_size: int=32):
        '''Train the model with the data of a single episode.
        DEPRECATED. is better to play some episodes and then call train_model:
        it shuffle the data causing a faster convergence'''
        
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


        ((states_tensor, policies_tensor,
        action_taken_tensor, zs_tensor),
        (valid_states_tensor, valid_policies_tensor,
        valid_action_taken_tensor, valid_zs_tensor)) = self.sample((state_matrix, policy_matrix,
                                                    action_takens_matrix, z_matrix),
                                                    0.85)

        with tqdm(total=states_tensor.shape[0] // batch_size,
                desc="First episode") as pbar:
            
            for batch_start in range(0, policies_tensor.shape[0], batch_size):
                batch_state = states_tensor[batch_start:batch_start + batch_size]
                batch_policy = policies_tensor[batch_start:batch_start + batch_size]
                batch_z = zs_tensor[batch_start:batch_start + batch_size]
                batch_action_taken = action_taken_tensor[batch_start:batch_start + batch_size]

                loss, v_loss, p_loss = self.train_batch(batch_state, batch_policy, batch_z, batch_action_taken)

                pbar.set_description(f"Loss: {loss:.5e}, Value: {v_loss:.5e}, Policy: {p_loss:.5e}")
                pbar.update()

        loss_sum = 0
        p_sum = 0
        v_sum = 0
        for k, batch_start in enumerate(range(0, valid_policies_tensor.shape[0], batch_size)):
            batch_state = valid_states_tensor[batch_start:batch_start + batch_size]
            batch_policy = valid_policies_tensor[batch_start:batch_start + batch_size]
            batch_z = valid_zs_tensor[batch_start:batch_start + batch_size]
            batch_action_taken = valid_action_taken_tensor[batch_start:batch_start + batch_size]

            v_pred, p_pred = self.model(batch_state)
            loss , v_loss, p_loss = self.loss_fn(batch_policy,
                                    batch_z,
                                    v_pred,
                                    p_pred)
            loss_sum += loss
            p_sum += p_loss
            v_sum += v_loss
        print(f"valid loss: {loss_sum / (k + 1):.5e}, "
              f"valid value Loss: {v_sum / (k + 1):.5e}, "
              f"valid policy loss: {p_sum / (k + 1):.5e}")

    def train_batch(self,
                    batch_state,
                    batch_policy,
                    batch_z,
                    batch_action_taken):
        '''Compute the loss and update the weights with the data for a single batch.
        Simmetries - the board can be rotated by 90, 180, 270 degree'''
        for _ in range(4):
            with tf.GradientTape() as tape:
                v_pred, p_pred = self.model(batch_state, training=True)
                loss , v_loss, p_loss = self.loss_fn(batch_policy,
                                        batch_z,
                                        v_pred,
                                        p_pred)

            batch_state = tf.image.rot90(batch_state)
            batch_policy = tf.image.rot90(batch_policy)
            
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss, v_loss, p_loss


    def get_last_samples(self, number_of_samples: int=100_000,
                         max_episode_length: int=100
                         ) -> tuple[list[tf.Tensor],
                                    list[tf.Tensor],
                                    list[tf.Tensor],
                                    list[tf.Tensor]]:
        ''' Load from disk the data of previos matches.
        
        number_of_samples : number of steps to load from memory
        max_episode_length : load steps only if part of a match no longer than this value (load some longer matches with score -0.1)'''
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
            print(f"loaded {filename}")
            file_state, file_policy, file_actions, file_actions_taken, file_z = zip(*file_data)
            
            cutted_file_state = []
            cutted_file_policy = []
            cutted_file_actions = []
            cutted_file_actions_taken = []
            cutted_file_z = []
            
            start = 0
            end = 0
            z_to_check = file_z[0]
            for z in file_z:
                if z == z_to_check:
                    end += 1
                else:
                    if end - start <= max_episode_length:
                        cutted_file_state.extend(file_state[start:end])
                        cutted_file_policy.extend(file_policy[start:end])
                        cutted_file_actions.extend(file_actions[start:end])
                        cutted_file_actions_taken.extend(file_actions_taken[start:end])
                        cutted_file_z.extend(file_z[start:end])
                    elif end - start <= max_episode_length * 2:
                        cutted_file_state.extend(file_state[start:start + max_episode_length])
                        cutted_file_policy.extend(file_policy[start:start + max_episode_length])
                        cutted_file_actions.extend(file_actions[start:start + max_episode_length])
                        cutted_file_actions_taken.extend(file_actions_taken[start:start + max_episode_length])
                        cutted_file_z.extend([-0.1 for _ in range(max_episode_length)])
                    start = end
                    z_to_check = z

            data_loaded['state'].extend(cutted_file_state)
            data_loaded['policy'].extend(cutted_file_policy)
            data_loaded['actions'].extend(cutted_file_actions)
            data_loaded['actions_taken'].extend(cutted_file_actions_taken)
            data_loaded['z'].extend(cutted_file_z)

            if len(data_loaded['z']) >= number_of_samples:
                print(len(data_loaded['z']))
                
                data_loaded['state'] = data_loaded['state'][:number_of_samples]
                data_loaded['policy'] = data_loaded['policy'][:number_of_samples]
                data_loaded['actions'] = data_loaded['actions'][:number_of_samples]
                data_loaded['actions_taken'] = data_loaded['actions_taken'][:number_of_samples]
                data_loaded['z'] = data_loaded['z'][:number_of_samples]
                break
        print(len(data_loaded['z']))
        state_matrix = tf.stack(data_loaded['state'], axis=0)
        policy_matrix = tf.stack([policy_to_policy_matrix(p, a)
                         for p, a in zip(data_loaded['policy'],
                                         data_loaded['actions'])], axis=0)
        
        action_takens_matrix = tf.stack(actions_to_indexes(data_loaded['actions_taken']), axis=0)
        z_matrix = tf.constant(data_loaded['z'], shape=(len(data_loaded['z']), 1))
        policy_matrix = tf.reshape(policy_matrix, (-1, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE * 2))
        return state_matrix, policy_matrix, action_takens_matrix, z_matrix



    def train_test_split(self, samples: tuple, split: float) -> tuple:
        '''Split the data with train and validation'''
        states, policies, action_takens, zs = samples
        num_element = int(states.shape[0] * split)
        mask = [0 for _ in range(states.shape[0] - num_element)] + [1 for _ in range(num_element)]
        random.shuffle(mask)
        valid_mask = [(m + 1) % 2 for m in mask]
        
        state_tensor = tf.boolean_mask(states, mask)
        policies_tensor = tf.boolean_mask(policies, mask)
        action_taken_tensor = tf.boolean_mask(action_takens, mask)
        zs_tensor = tf.boolean_mask(zs, mask)

        val_state_tensor = tf.boolean_mask(states, valid_mask)
        val_policies_tensor = tf.boolean_mask(policies, valid_mask)
        val_action_taken_tensor = tf.boolean_mask(action_takens, valid_mask)
        val_zs_tensor = tf.boolean_mask(zs, valid_mask)
        
        return (state_tensor, policies_tensor, action_taken_tensor, zs_tensor), (val_state_tensor, val_policies_tensor, val_action_taken_tensor, val_zs_tensor)

    def shuffle(self, samples: tuple):
        ''' shuffle the data'''
        states, policies, action_takens, zs = samples
        indices = tf.range(start=0, limit=states.shape[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        return (tf.gather(states, shuffled_indices),
                tf.gather(policies, shuffled_indices),
                tf.gather(action_takens, shuffled_indices),
                tf.gather(zs, shuffled_indices))




    def sample(self, samples: tuple, final_size: float) -> tuple:
        '''train test split and shuffle of data. DEPRECATED: use train_test_split and shuffle'''
        states, policies, action_takens, zs = samples
        num_element = int(states.shape[0] * final_size)

        mask = [0 for _ in range(states.shape[0] - num_element)] + [1 for _ in range(num_element)]
        random.shuffle(mask)
        valid_mask = [(m + 1) % 2 for m in mask]

        state_tensor = tf.boolean_mask(states, mask)
        policies_tensor = tf.boolean_mask(policies, mask)
        action_takens_tensor = tf.boolean_mask(action_takens, mask)
        zs_tensor = tf.boolean_mask(zs, mask)

        valid_state_tensor = tf.boolean_mask(states, valid_mask)
        valid_policies_tensor = tf.boolean_mask(policies, valid_mask)
        valid_action_takens_tensor = tf.boolean_mask(action_takens, valid_mask)
        valid_zs_tensor = tf.boolean_mask(zs, valid_mask)

        indices = tf.range(start=0, limit=state_tensor.shape[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        return (tf.gather(state_tensor, shuffled_indices),
                tf.gather(policies_tensor, shuffled_indices),
                tf.gather(action_takens_tensor, shuffled_indices),
                tf.gather(zs_tensor, shuffled_indices)), (
            valid_state_tensor,
            valid_policies_tensor,
            valid_action_takens_tensor,
            valid_zs_tensor)
    
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
            except Exception as e:
                print(str(e))
                time.sleep(retry_time)
        return data

    def save_cache(self, cache: list[list], cache_mb_size:int=20):
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
        self.sync_file_op(cache_filename, "wb",
                          lambda f: pickle.dump(cache_data, f))
