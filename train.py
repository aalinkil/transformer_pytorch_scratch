import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm
import warnings

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

## Change defaults of variables in this function
## This function allows you to change settings in Command line interface
def getting_vars():

    ## Set if you're using your windows or mac machine (data directory location)
    machine = 'w'     # m for mac, w for windows

    ## Set if setting the model in training mode or inference mode
    execution = 't'     # t for train, s for sample

    ## Model Variables
    input_splits = 3
    num_heads = 6
    num_encoders = 

    ## Hyperparameters
    training_batch_size = 8
    training_epochs = 50
    learning_rate = 1e-3

    ## Adding final sigmoid layer to U-Net Model
    sigmoid = 'y'

    parser = argparse.ArgumentParser()
    parser.add_argument('-machine', type = str, default = machine, choices=['w', 'm'])
    parser.add_argument('-execution', type = str, default = execution, choices=['t', 's'], 
                        help = 't for train, or s for sampling the model')
    parser.add_argument('-in_channels', type = int, default = in_channels, 
                        help = '# of channels present in the input data')
    parser.add_argument('-num_classes', type = int, default = num_classes, 
                        help = '# of output classes from the U-net model')
    parser.add_argument('-num_filters', type = int, default = num_filters, 
                        help = '# of filters you want the first layer of the U-net model to have')
    parser.add_argument('-training_batch_size', type = int, default = training_batch_size, 
                        help = 'Batch Size during Training and Validation')
    parser.add_argument('-training_epochs', type = int, default = training_epochs, 
                        help = '# of epochs you want to train the model')
    parser.add_argument('-learning_rate', type = float, default = learning_rate, 
                        help = 'learning rate')
    parser.add_argument('-sigmoid_end', type = str, default = sigmoid, choices=['y', 'n'],
                        help = 'If final layer of Unet should be sigmoid or not (y/n)')

    args = parser.parse_args()
    return args
