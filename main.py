import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# Importing user made libraries
from dataset import *
from model import build_transformer
from utils import *
from config import *
from input_funcs import *

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

import argparse, os

## Change defaults of variables in this function
## This function allows you to change settings in Command line interface



if __name__ == "__main__":

    # select device
    device = get_device()
    device = torch.device(device)

    # Get arguments from user
    getting_vars()

    ## Visually see what the variables are set to in the Command line interface
    ### DO NOT CHANGE THE VARIABLES IN HERE. CHANGE IT IN THE config.py file 
    print(f"Machine: {machine}")
    print(f"Execution: {execution}")
    print(f"Input Splits: {input_splits}")
    print(f"Number of Heads: {num_heads}")
    print(f"Number of Encoder/Decoder Blocks: {num_enc_dec_blocks}")
    print(f"Embeddings Dimension: {emb_dim}")
    print(f"Training Batch Size: {training_batch_size}")
    print(f"Training Epochs: {training_epochs}")
    print(f"Learning Rate: {learning_rate}")


    # Data Paths
    current_dir = os.getcwd()

    ## Make sure the weights folder exists
    Path(f"{datasource}_{model_folder}").mkdir(parents=True, exist_ok=True)

    # Loading Data
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_ds(training_batch_size)

    # Create Model
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size(), seq_len, seq_len, d_model=emb_dim)

    # Create Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    """
    - nn.CrossEntropyLoss: This is a loss function provided by PyTorch (torch.nn.CrossEntropyLoss). It is commonly used for 
    classification problems, where it combines nn.LogSoftmax and nn.NLLLoss (Negative Log Likelihood Loss) into one 
    single class. It calculates the cross-entropy loss between the predicted probabilities and the true labels.

    - ignore_index: This parameter specifies a target value that should be ignored when computing the loss. It is useful in 
    tasks like sequence-to-sequence models where padding tokens are used to fill sequences to a fixed length.

    - tokenizer_src.token_to_id('[PAD]'): This function call converts the padding token [PAD] into its corresponding token ID. 
    In this case, ignore_index is set to this padding token ID. When the loss function encounters this token in the target data,
     it will ignore it when computing the loss. This prevents the model from learning from the padding tokens.

    -label_smoothing=0.1: label_smoothing: This parameter is used to apply label smoothing. Label smoothing is a 
    regularization technique that softens the target labels. Instead of using a one-hot encoded vector where the true class
    has a value of 1 and all other classes have a value of 0, label smoothing assigns a small probability (in this case, 0.1)
    to all other classes, while reducing the probability of the true class. With label_smoothing=0.1, the true label 
    is smoothed to 1 - 0.1 = 0.9, and the remaining 0.1 is distributed among the other classes. This technique can help to 
    prevent overfitting and improve model generalization."""

    # select optimizer for training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-9)

    if execution == 't':
        train_loop(model= model, loss_fn= loss_fn, optimizer= optimizer,)

    if execution == 's':
        pass