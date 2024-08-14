# Import torch libraries
import torch
from torch.utils.tensorboard import SummaryWriter

# Import User Created libraries
from config import *

from pathlib import Path
from tqdm import tqdm
import os
from datetime import datetime


def get_device():
    ### Finds and sets cuda, mac processors, or regular cpu as device

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")


def get_weights_file_path(epoch: str):
    model_folder = f"{datasource}_{model_folder}"
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path():
    model_folder = f"{datasource}_{model_folder}"
    model_filename = f"{model_basename}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
    

def train_loop(model, loss_fn, optimizer, ):
    ### Model training takes place in this function

    initial_epoch = 0
    global_step = 0
    preload = PRELOAD

    # model filename based on if any previous file was found
    model_filename = latest_weights_file_path() if preload == 'latest' else get_weights_file_path(preload) if preload else None

    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # Setup TensorBoard
    curr_dt = datetime.now()
    curr_dt = curr_dt.strftime("%d-%m-%Y %H-%M")
    writer = SummaryWriter(log_dir=f'runs/T_training/{curr_dt}') # creates folder based on current date time
    writer.add_graph(model) # check if images.to(device) works if this doesn't work at all
    writer.close()

    ### CONTINUE WITH EPOCH LOOP
