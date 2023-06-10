import json
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

from games.tablut.players.reinforce import Model as RNModel
from games.tablut.players.alpha_zero import Model as AZNModel
from games.tablut.players.alpha_reinforce import Model as ARNModel
from games.tablut_simple.players.reinforce import Model as RSModel
from games.tablut_simple.players.alpha_zero import Model as AZSModel
from games.tablut_simple.players.alpha_reinforce import Model as ARSModel

app = FastAPI()

# Define the request model
class InputData(BaseModel):
    model_path: str
    data: list

class TrainInputData(BaseModel):
    model_path: str
    data: bytes

class SaveFileInputData(BaseModel):
    file_path: str
    config_file: str

# Timer in second to predict the received data
queue_time = 0.03

# How many train request received before actually train
train_size = 10

loaded_models = {}
batch_data: dict[str, dict[int, tf.Tensor]] = {}
batch_train_data = dict[str, dict]
batch_result = {}
last_session_id = 0
session_id_to_model_path = {}
prediction_ready = {}
training_event = {}
train_episode_data = {}
config_file_semaphore = asyncio.Semaphore(1)

models_paths = {
    'simple_reinforce': ("models/simple_reinforce", RSModel),
    'simple_alpha_zero': ("models/simple_alpha_zero", AZSModel),
    'simple_alpha_reinforce': ("models/simple_alpha_reinforce", ARSModel),

    'reinforce': ("models/reinforce", RNModel),
    'alpha_zero': ("models/alpha_zero", AZNModel),
    'alpha_reinforce': ("models/alpha_reinforce", ARNModel),
}


train_requests = {
    'simple_reinforce': 0,
    'simple_alpha_zero': 0,
    'simple_alpha_reinforce': 0,
    'reinforce': 0,
    'alphazero': 0,
    'alpha_reinforce': 0,
}

@app.on_event("startup")
@repeat_every(seconds=queue_time)
async def process_batch_automatic_caller() -> None:
    for model_path in batch_data.keys():
        if ((model_path not in training_event or 
             training_event[model_path].is_set()) and
            len(batch_data[model_path]) > 0):
            process_batch(model_path)

def process_batch(model_path):
    global batch_data, batch_result, prediction_ready
    
    # Convert the batch data to a NumPy array
    data_keys, data_values = list(zip(*batch_data[model_path].items()))

    batch_tensor = tf.concat(data_values, axis=0)
    # Make predictions using the TensorFlow model
    predictions = list(zip(*loaded_models[model_path].predict(batch_tensor)))
    
    for key, b_pred in zip(data_keys, predictions):
        batch_result[key] = b_pred
        del batch_data[model_path][key]

        prediction_ready[key].set()
        del prediction_ready[key]
    
@app.get("/train/{model}")
async def train(model: str):
    global train_requests, models_paths, train_size, loaded_models
    if model in models_paths:
        train_requests[model] += 1
        if len(train_requests[model]) > train_size:
            if model not in training_event:
                training_event[model] = asyncio.Event()
            training_event[model].clear()
            path, model_class = models_paths[model]
            model = model_class(path)
            model.train_model()
            model.save_model()
            loaded_models[path] = model.model
            training_event[model].set()
        return Response(status_code=200)
    return Response(status_code=202)

@app.post("/train_episode")
async def train_episode(request: Request, background_tasks: BackgroundTasks):
    global train_episode_data
    model_path = request.query_params.get("model_path")
    data: bytes = await request.body()
    if model_path not in train_episode_data:
        train_episode_data[model_path] = []
    train_episode_data[model_path].extend(pickle.loads(data))

    if len(train_episode_data[model_path]) > train_size:
        print("add train to background")
        if model_path in training_event:
            await training_event[model_path].wait()
        background_tasks.add_task(train_model_episodes, model_path)
        return Response(status_code=200)
    return Response(status_code=202)

async def train_model_episodes(model_path):
    global train_episode_data, loaded_models, training_event
    if model_path not in training_event:
        training_event[model_path] = asyncio.Event()
    else:
        await training_event[model_path].wait()
    training_event[model_path].clear()

    path, model_class = models_paths[model_path]
    data_for_training = train_episode_data[model_path]
    del train_episode_data[model_path]
    model = model_class(path)
    model.train_episode(data_for_training)
    model.save_model()

    loaded_models[path] = model.model

    training_event[model_path].set()


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

    my_session_id = last_session_id
    if data.model_path not in batch_data:
        loaded_models[data.model_path] = tf.keras.models.load_model(models_paths[data.model_path][0] + "/last_model")
        batch_data[data.model_path] = {}

    
    session_id_to_model_path[my_session_id] = data.model_path
    batch_data[data.model_path][my_session_id] = tf.convert_to_tensor(data.data)
    my_event = asyncio.Event()
    prediction_ready[my_session_id] = my_event
    while True:
        await my_event.wait()
        return Response(pickle.dumps(get_result(my_session_id)), status_code=200)