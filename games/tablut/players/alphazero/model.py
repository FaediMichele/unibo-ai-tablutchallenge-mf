import tensorflow as tf
from tensorflow import keras
from keras import layers
from tqdm import tqdm
import numpy as np
import json
import os
import pickle

from .util import policy_to_policy_matrix, PREVIOUS_STATE_TO_MODEL, SAVE_ITERATIONS

loaded_model: tuple[keras.Model, keras.optimizers.Optimizer] = None

class ModelUtil:
    # TODO do a reg net instead
    @staticmethod
    def create_model(board_size: tuple[int, int]) -> keras.Model:
        # Made with ChatGPT with few corrections ;) - inspired by RegNetY
        # Define the input layer

        # history + current_state + state_player + playing player
        input_shape = (board_size[0], board_size[1], PREVIOUS_STATE_TO_MODEL + 3)
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
        value = layers.Dense(1, 'linear')(value)

        # Extract the policy
        x = layers.Conv2D(81, 1, 1, 'same', activation='swish')(x)
        x = keras.layers.Reshape(target_shape=(9, 9, 9, 9))(x)
        policy = layers.Conv3D(9, 3, 1, 'same', activation='softmax')(x)
        # Define the model
        model = tf.keras.Model(inputs=inputs, outputs=[value, policy])

        return model
    

    @staticmethod
    def create_model_old(board_size: tuple[int, int]
                     ) -> tuple[keras.Model, keras.optimizers.Optimizer]:
        # Old model - my implementation but not anymore supported
        # 1 channel for the current player and 1 for the current state
        input_shape = (PREVIOUS_STATE_TO_MODEL + 2, board_size[0], board_size[1])
        input_layer = keras.Input(input_shape) 
        x = layers.Conv2D(64, 3, 1, 'same', activation='swish')(input_layer)
        x = layers.Conv2D(64, 3, 1, 'same', activation='swish')(x)
        x = layers.Conv2D(81, 3, 1, 'same', activation='swish',
                          kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x_p = layers.Conv3D(9, 3, 1, 'same', activation='swish',
                          kernel_regularizer=keras.regularizers.l2(0.01))(
            tf.reshape(x, (-1,9,9,9,9)))
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
        
        global loaded_model
        if loaded_model is not None:
            return loaded_model
        
        current_model_path = ModelUtil.get_last_model_path(path)

        optimizer = keras.optimizers.Adam()
        if current_model_path is not None:
            model = keras.models.load_model(current_model_path, compile=False)
        else:
            model = ModelUtil.create_model(board_size)

        loaded_model = (model, optimizer)
        return loaded_model

    
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
    def player_wins(player: str, path: str='alpha_zero'):
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        config['wins'].append(player)
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f)

    @staticmethod
    def predict(model: tuple[keras.Model, keras.optimizers.Optimizer], data):
        data = tf.expand_dims(data, axis=0)
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
                    epochs: int=1,
                    winner: str=-1):
        if winner >= 0:
            ModelUtil.player_wins(winner, cache_folder)
        model, optimizer = model
        if cache is not None:
            ModelUtil.save_cache(cache, cache_folder)

        states_tensor, policies_tensor, zs_tensor  = ModelUtil.sample(
            ModelUtil.get_last_samples(cache_folder,
                                       batch_size * step_for_epoch), 0.3
        )
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
            
            print(f"Done epoch {e+1}/{epochs}")

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
    def save_cache(cache: list[list], folder_path: str,
                   cache_mb_size:int=20):
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