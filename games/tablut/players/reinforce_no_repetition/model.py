from games.tablut.game import Action
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow.keras.backend as K
from tqdm import tqdm
import numpy as np
import json
import os
import pickle
import datetime
import random
from more_itertools import ichunked
from itertools import repeat

from .util import boolean_mask_from_coordinates, no_repetition_actions, actions_to_indexes, PREVIOUS_STATE_TO_MODEL, SAVE_ITERATIONS


class Model:
    # TODO do a reg net instead
    def __init__(self, path: str='reinforce_no_repetition', model_path:str=None) -> None:
        self.model: keras.Model = None
        self.optimizer: keras.optimizers.Optimizer = None
        self.path = path
        self.first_game = True
        self.load_model(model_path)

    def load_model(self, model_path:str=None):
        if self.optimizer is None:
            self.optimizer = keras.optimizers.Adam()

        if model_path is None:
            model_path = self.get_last_model_path()
        if model_path is not None:
            model = keras.models.load_model(model_path, compile=False)
        else:
            model = self.create_model()
        self.model = model

    def save_model(self, path: str=None):
        if path is None:
            path = self.path
        config = self.load_config_file()
        current_model_path = os.path.join(path, config['current'])
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
        config_path = os.path.join(self.path, 'config.json')
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            config = {
                    'current': 'last_model',
                    'wins': [],
                    'total_seconds': 0,
                    'last_update': datetime.datetime.now().strftime(
                                                    "%Y-%m-%d %H:%M:%S")
            }
            with open(config_path, 'w') as f:
                json.dump(config, f)
            return config
        
    def save_config_file(self, config: dict):
        config_path = os.path.join(self.path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    
    def player_wins(self, player: str, remaining_moves: int):
        config = self.load_config_file()
        config['wins'].append(f"{player}-{-remaining_moves}")
        self.save_config_file(config)


    def predict(self, data):
        data = tf.expand_dims(data, axis=0)
        value, policy = self.model.predict(data, verbose=0)
        policy = tf.nn.softmax(policy)
        if tf.reduce_any(tf.math.is_nan(policy)):
            print(data, "\n", policy)
        return value[0][0], policy[0]

    
    def loss_fn(self, states, z, v_pred, p_pred, actions_tensor):
        # stop_gradient allow to use one network both for policy and value.
        # Without stop_gradient the update of the policy also update the value
        # creating a sort of feedback loop causing the network going to NaN
        d = tf.stop_gradient(z - v_pred)
        state_value_loss = -tf.reduce_mean(v_pred * d)
        
        
        p_pred_flat = tf.reshape(p_pred, [p_pred.shape[0], 9 * 9 * 18])
        p_pred_flat = tf.nn.log_softmax(p_pred_flat, axis=1)
        p_softmax = tf.reshape(p_pred_flat, [p_pred.shape[0], 9, 9, 18])

        #p_softmax = tf.keras.activations.softmax(p_pred, axis=[1, 2, 3])
        #print(p_softmax)
        # p_softmax = tf.math.log(p_softmax + 0.1)
        #p_softmax = tf.nn.log_softmax(p_pred, axis=[1, 2, 3])
        r = tf.expand_dims(tf.range(actions_tensor.shape[0]), axis=1)
        indices = tf.concat([r, actions_tensor], axis=1)
        selected_log_policy = tf.gather_nd(p_softmax, indices)
        
        
        policy_loss = -tf.reduce_mean(selected_log_policy * d)


        # Discourage action that leads to repeated states
        action_repeated = [no_repetition_actions(states[idx]) for idx in range(z.shape[0])]
        mask = boolean_mask_from_coordinates(action_repeated)
        repeated_actions = tf.boolean_mask(p_softmax, mask)


        return state_value_loss + policy_loss + repeated_actions, policy_loss, state_value_loss, tf.reduce_mean(d)
    
    
    def train_episode(self, state_action: list[tuple[tf.Tensor, Action]],
                      z: list[float], batch_size:int=32) -> float:
        
        data_shuffled = list(zip(state_action, z))
        random.shuffle(data_shuffled)
        state_action, z = zip(*data_shuffled)

        with tqdm(total=len(state_action) // batch_size,
                  desc="First episode") as pbar:
            p_sum = 0
            v_sum = 0
            d_sum = 0
            for k, batch in enumerate(ichunked(zip(z, state_action),
                                               batch_size)):
                z_batch, batch = zip(*batch)
                states_batch, actions_batch = zip(*batch)
                p_loss, v_loss, d_loss = self.train_batch(states_batch,
                                                  actions_batch,
                                                  z_batch)
                p_sum += p_loss
                v_sum += v_loss
                d_sum += d_loss
                pbar.set_description(f"Value Loss: {v_sum / (k + 1):.5e}, "
                                     f"Policy loss: {p_sum / (k + 1):.5e}, "
                                     f"D value: {d_sum / (k + 1):.5e}")
                pbar.update()

    def train_model(self,
                    batch_size: int=32,
                    step_for_epoch: int=30,
                    epochs: int=1):

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
                                                      actions_batch, zs_batch)
                    p_sum += p_loss
                    v_sum += v_loss
                    d_sum += d_loss
                    pbar.set_description(f"Value Loss: {v_sum / (k + 1):.5e}, "
                                        f"Policy loss: {p_sum / (k + 1):.5e}, "
                                        f"D value: {d_sum / (k + 1):.5e}")
                    pbar.update()
            print(f"Done epoch {e+1}/{epochs}")


    def train_batch(self, states: list[tf.Tensor], actions: list[Action],
                    z: list[float]):
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

    def get_last_samples(self, number_of_games: int=1000
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

        
        data_loaded = []
        for filename, _ in sorted(file_indexes, key=lambda x: x[1],
                                  reverse=True):
            with open(os.path.join(self.path, filename), "rb") as f:
                file_games = pickle.load(f)

            data_loaded.extend(file_games)
            if len(data_loaded) >= number_of_games:
                data_loaded = data_loaded[:number_of_games]
                break
            print(filename)
        return data_loaded
    

    def sample(self, samples: tuple, final_size: float) -> tuple:
        states, zs = samples
        num_element = int(states.shape[0] * final_size)
        sum_prob = states.shape[0] * (states.shape[0] + 1) / 2
        choiches = np.random.choice(range(0, states.shape[0]), num_element,
                         p=[i/sum_prob for i in range(1, states.shape[0]+1)])
        mask = [i in choiches for i in range(0, states.shape[0])]
        return (tf.boolean_mask(states, mask),
                tf.boolean_mask(zs, mask))
    
    def get_last_model_path(self) -> str | None:
        config = self.load_config_file()
        current_model_path = os.path.join(self.path, config['current'])
        if os.path.isdir(current_model_path):
            return current_model_path
        return None


    def save_cache(self, cache: list[tuple[tf.Tensor, float]],
                   cache_mb_size:int=20):
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
                with open(latest_cache_file, "rb") as f:
                    cache_data = pickle.load(f)

                # Otherwise, append to the latest cache file
                cache_filename = latest_cache_file
        
        # Append the new data to the cache
        cache_data.extend(cache)
        with open(cache_filename, "wb") as f:
            pickle.dump(cache_data, f)


    def create_model(self) -> keras.Model:
        input_shape = (9, 9, PREVIOUS_STATE_TO_MODEL + 2)
        policy_shape = (9, 9, 18)
        
        input_layer = keras.Input(shape=input_shape)
        
        policy_output = self.create_policy_network(input_shape, policy_shape)(input_layer)
        state_value_output = self.create_value_network(input_shape)(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=[state_value_output, policy_output])
        return model
        

    def create_policy_network(self, input_shape, policy_shape):
        # Made with ChatGPT with few corrections ;)
        input_layer = keras.Input(shape=input_shape)
        x = layers.Conv2D(64, (3, 3), activation='relu')(input_layer)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        
        dense_shape = policy_shape[0] * policy_shape[1] * policy_shape[2]
        x = layers.Dense(dense_shape, activation='linear')(x)
        x = layers.Reshape(policy_shape)(x)
        

        return tf.keras.Model(inputs=input_layer, outputs=x)

    def create_value_network(self, input_shape):
        # Made with ChatGPT with few corrections ;)
        input_layer = keras.Input(shape=input_shape)
        x = layers.Conv2D(64, (3, 3), activation='relu')(input_layer)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(1, activation='linear')(x)
        

        return tf.keras.Model(inputs=input_layer, outputs=x)