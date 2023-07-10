import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import BackgroundTasks, FastAPI, Response, Request
from fastapi.responses import StreamingResponse
from fastapi_utils.tasks import repeat_every
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import numpy as np
import asyncio
import pickle
import datetime

from games.tablut.players.reinforce import Model as TablutReinforceModel
from games.tablut.players.alpha_zero import Model as TablutAlphaZeroModel
from games.tictactoe.players.reinforce import Model as TictactoeReinforceModel
from games.tictactoe.players.alpha_zero import Model as TictactoeAlphZeroModel

app = FastAPI()

# Define the request model
class InputData(BaseModel):
    model_path: str
    data: list
    old_model: bool

class TrainInputData(BaseModel):
    model_path: str
    data: bytes

class SaveFileInputData(BaseModel):
    file_path: str
    config_file: str

# Timer in second to predict the received data
queue_time = 0.001

# How many train request received before actually train
train_size = 200
train_batch_size = 128

loaded_models = {}
batch_data: dict[str, dict[int, tf.Tensor]] = {}
batch_train_data = dict[str, dict]
batch_result = {}
last_session_id = 0
session_id_to_model_path = {}
prediction_ready = {}
training_event = {}
train_episode_data = {}
predicting_phase = asyncio.Event()
predicting_phase.set()


# Due to implementation errors the key and the string must be equals
models_paths = {

    'models/reinforce_tictactoe': ("models/reinforce_tictactoe", TictactoeReinforceModel),
    'models/alpha_zero_tictactoe': ("models/alpha_zero_tictactoe", TictactoeAlphZeroModel),
    'models/reinforce': ("models/reinforce", TablutReinforceModel),
    'models/alpha_zero': ("models/alpha_zero", TablutAlphaZeroModel),
}

train_requests = {
    'models/reinforce_tictactoe': 0,
    'models/alpha_zero_tictactoe': 0,
    'models/reinforce': 0,
    'models/alpha_zero': 0,
}


@app.on_event("startup")
def load_models():
    global loaded_models
    for key, (_, model_class) in models_paths.items():
        loaded_models[f'{key}_old'] = model_class(old_model=True).model
        loaded_models[key] = model_class().model
        print(key, "loaded")

@app.on_event("startup")
@repeat_every(seconds=queue_time)
async def process_batch_automatic_caller() -> None:
    await predicting_phase.wait()
    predicting_phase.clear()
    await process_batches()
    predicting_phase.set()


async def process_batches():
    global test_loaded_models
    for model_path in batch_data.keys():
        if (model_path not in training_event or 
            training_event[model_path].is_set()):

            if len(batch_data[model_path]) > 0:
                # print("processing", model_path)
                process_batch(model_path)

def process_batch(model_path):
    global batch_data, batch_result, prediction_ready, loaded_models
    
    # Convert the batch data to a NumPy array
    data_keys, data_values = list(zip(*batch_data[model_path].items()))

    batch_tensor = tf.concat(data_values, axis=0)
    # Make predictions using the TensorFlow model
    with tf.device('/device:GPU:0'):
        predictions = list(zip(*loaded_models[model_path].predict(batch_tensor, verbose=-1)))

    for key, b_pred in zip(data_keys, predictions):
        batch_result[key] = b_pred
        del batch_data[model_path][key]

        prediction_ready[key].set()
        del prediction_ready[key]

@app.post("/train_episode")
async def train_episode(request: Request, background_tasks: BackgroundTasks):
    global train_episode_data, train_requests
    model_path = request.query_params.get("model_path")
    data: bytes = await request.body()
    if model_path not in train_episode_data:
        train_episode_data[model_path] = []
    train_episode_data[model_path].extend(pickle.loads(data))
    train_requests[model_path] += 1

    if train_requests[model_path] > train_size:
        print("add train to background")
        if model_path in training_event:
            await training_event[model_path].wait()
        background_tasks.add_task(train_model_episodes, model_path)
        return Response(status_code=200)
    return Response(status_code=202)

async def train_model_episodes(model_path):
    global train_episode_data, loaded_models, training_event, train_requests
    if model_path not in training_event:
        training_event[model_path] = asyncio.Event()
    else:
        await training_event[model_path].wait()
    training_event[model_path].clear()

    path, model_class = models_paths[model_path]
    data_for_training = train_episode_data[model_path]
    del train_episode_data[model_path]
    model = model_class()
    model.model = loaded_models[path]
    model.train_model(epochs=2, step_for_epoch=1200, batch_size=train_batch_size)
    model.save_model()

    loaded_models[path] = model.model
    loaded_models[f"{path}_old"] = model_class(old_model=True).model

    train_requests[model_path] = 0
    training_event[path].set()


# Function used to reload batch_result.
# Using directly in predict does not reload the variable
def get_result(session_id):
    global batch_result

    if session_id in batch_result:
        res = batch_result[session_id]
        del batch_result[session_id]
        return res
    return False
    
@app.post("/predict")
async def predict(data: InputData):
    global last_session_id, batch_data, loaded_models, models_paths, prediction_ready
    last_session_id += 1

    
    if data.old_model:
        model_to_use = f'{data.model_path}_old'
    else:
        model_to_use = data.model_path

    my_session_id = last_session_id
    if model_to_use not in batch_data:
        batch_data[model_to_use] = {}
    
    session_id_to_model_path[my_session_id] = model_to_use
    batch_data[model_to_use][my_session_id] = tf.convert_to_tensor(data.data)
    my_event = asyncio.Event()
    prediction_ready[my_session_id] = my_event
    await my_event.wait()
    return Response(pickle.dumps(get_result(my_session_id)), status_code=200)
