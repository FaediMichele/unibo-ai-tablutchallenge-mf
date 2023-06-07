import tensorflow as tf
from tensorflow import keras
from keras import layers
from tqdm import tqdm
import numpy as np
import json
import os
import pickle
import datetime

from .util import policy_to_policy_matrix, PREVIOUS_STATE_TO_MODEL, SAVE_ITERATIONS

class Model:
    # TODO do a reg net instead
    def __init__(self, path: str='simple_alpha_zero') -> None:
        self.model: keras.Model = None
        self.optimizer: keras.optimizers.Optimizer = None
        self.path = path
        self.first_game = datetime.datetime.now()
        self.load_model()

    def load_model(self):
        if self.optimizer is None:
            self.optimizer = keras.optimizers.Adam()
        current_model_path = self.get_last_model_path()
        if current_model_path is not None:
            model = keras.models.load_model(current_model_path, compile=False)
        else:
            model = self.create_model()
        self.model = model

    
    def save_model(self):
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
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            config = {
                    'current': 'last_model',
                    'wins': [],
                    'total_seconds': 0,
                    'last_update': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(config_path, 'w') as f:
                json.dump(config, f)
            return config
        
    def save_config_file(self, config: dict):
        config_path = os.path.join(self.path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
    
    def player_wins(self, player: str):
        config = self.load_config_file()
        config['wins'].append(player)
        self.save_config_file(config)

    def predict(self, data):
        data = tf.expand_dims(data, axis=0)
        value, policy = self.model.predict(data, verbose=0)
        policy = tf.nn.softmax(policy)
        if tf.reduce_any(tf.math.is_nan(policy)):
            print(data, "\n", policy)
            raise ValueError('keras.model.predict returned value with NaN\n'
                             f"Input data: {data}\nPrediction: {value, policy}")
        return value[0][0], policy[0]

    @tf.function
    def loss_fn(self, pi, z, v, p):
        
        pi = tf.reshape(pi, (pi.shape[0],-1))
        p = tf.reshape(p, (p.shape[0],-1))
        mse = tf.reduce_mean(keras.losses.mean_squared_error(z, v))
        cross_entropy = tf.reduce_mean(keras.losses.categorical_crossentropy(pi, p))
        penalty = sum(self.model.losses)
        return mse + cross_entropy + penalty
    
    def train_model(self,
                    batch_size: int=32,
                    step_for_epoch: int=100,
                    epochs: int=1):

        last_samples = self.get_last_samples(batch_size * step_for_epoch)
        
        for e in range(epochs):
            states_tensor, policies_tensor, zs_tensor  = self.sample(
            last_samples, 0.25)
            for batch_start in tqdm(range(0, policies_tensor.shape[0],
                                          batch_size)):
                batch_state = states_tensor[batch_start:batch_start +\
                                                   batch_size]
                batch_policy = policies_tensor[batch_start:batch_start + batch_size]
                batch_z = zs_tensor[batch_start:batch_start + batch_size]

                with tf.GradientTape() as tape:
                    v_pred, p_pred = self.model(batch_state, training=True)
                    loss = self.loss_fn(batch_policy, batch_z, v_pred, p_pred)
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            
            print(f"Done epoch {e+1}/{epochs}")

    def get_last_samples(self, number_of_samples: int=100_000) -> tuple[list[tf.Tensor], list[tf.Tensor], list[tf.Tensor]]:
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
            'z': []
        }
        for filename, _ in sorted(file_indexes, key=lambda x: x[1],
                                  reverse=True):
            with open(os.path.join(self.path, filename), "rb") as f:
                file_data = pickle.load(f)

            file_state, file_policy, file_actions, file_z = zip(*file_data)
            data_loaded['state'].extend(file_state)
            data_loaded['policy'].extend(file_policy)
            data_loaded['actions'].extend(file_actions)
            data_loaded['z'].extend(file_z)
            if len(data_loaded['z']) >= number_of_samples:
                
                data_loaded['state'] = data_loaded['state'][:number_of_samples]
                data_loaded['policy'] = data_loaded['policy'][:number_of_samples]
                data_loaded['actions'] = data_loaded['actions'][:number_of_samples]
                data_loaded['z'] = data_loaded['z'][:number_of_samples]
                break

        state_matrix = tf.stack(data_loaded['state'], axis=0)
        policy_matrix = tf.stack([policy_to_policy_matrix(p, a)
                         for p, a in zip(data_loaded['policy'],
                                         data_loaded['actions'])], axis=0)
        z_matrix = tf.constant(data_loaded['z'], shape=(len(data_loaded['z']), 1))
        return state_matrix, policy_matrix, z_matrix

    def sample(self, samples: tuple, final_size: float) -> tuple:
        states, policies, zs = samples
        num_element = int(states.shape[0] * final_size)
        sum_prob = states.shape[0] * (states.shape[0] + 1) / 2
        choiches = np.random.choice(range(0, states.shape[0]), num_element,
                         p=[i/sum_prob for i in range(1, states.shape[0]+1)])
        mask = [i in choiches for i in range(0, states.shape[0])]
        return (tf.boolean_mask(states, mask),
                tf.boolean_mask(policies, mask),
                tf.boolean_mask(zs, mask))
    
    def get_last_model_path(self) -> str | None:
        config = self.load_config_file()
        current_model_path = os.path.join(self.path, config['current'])
        if os.path.isdir(current_model_path):
            return current_model_path
        return None

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
                with open(latest_cache_file, "rb") as f:
                    cache_data = pickle.load(f)

                # Otherwise, append to the latest cache file
                cache_filename = latest_cache_file
        
        # Append the new data to the cache
        cache_data.extend(cache)
        with open(cache_filename, "wb") as f:
            pickle.dump(cache_data, f)

    def create_model(self) -> keras.Model:
        # Made with ChatGPT with few corrections ;) - inspired by RegNetY
        # Define the input layer

        # history + current_state + state_player
        input_shape = (7, 7, PREVIOUS_STATE_TO_MODEL + 2)
        inputs = tf.keras.Input(shape=input_shape)

        # Define the base width, slope, and expansion factor parameters for RegNetY-200MF
        w_0 = 24
        w_a = 24.48
        w_m = 2.49
        d = 13

        # Compute the number of channels for each block group
        def compute_channels(n_blocks, w_prev):
            w = w_prev
            channels = []
            for i in range(n_blocks):
                w_next = int(round(w * w_a / w_0 ** (w_m / d)))
                channels.append(w_next)
                w = w_next
            return channels

        # Compute the channels for each block group in RegNetY-200MF
        channels = compute_channels(n_blocks=3, w_prev=w_0)

        # First stage
        x = layers.Conv2D(filters=channels[0], kernel_size=3,
                          padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Second stage
        for i in range(1):

            # Compute the channels for this block group
            n_channels = channels[i]

            # Residual path
            identity = x

            # BottleNeckBlock
            x = layers.Conv2D(filters=n_channels, kernel_size=1, strides=1)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters=n_channels, kernel_size=3, strides=1,
                              padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters=n_channels * 4, kernel_size=1, strides=1)(x)
            x = layers.BatchNormalization()(x)

            # Skip connection
            if identity.shape[-1] != x.shape[-1]:
                identity = layers.Conv2D(filters=n_channels * 4,
                                         kernel_size=1, strides=1)(identity)
                identity = layers.BatchNormalization()(identity)

            x = layers.Add()([x, identity])
            x = layers.ReLU()(x)
        
        # Extract the value
        value = layers.Conv2D(16, 3, 1, activation='swish')(x)
        value = layers.Flatten()(value)
        value = layers.Dense(16, 'swish')(value)
        value = layers.Dense(1, 'sigmoid')(value)
        value = layers.Lambda(lambda x: x * 2 - 1)(value)

        # Extract the policy
        policy = layers.Conv2D(14, 1, 1, 'same', activation='softmax')(x)
        # Define the model
        model = tf.keras.Model(inputs=inputs, outputs=[value, policy])

        return model