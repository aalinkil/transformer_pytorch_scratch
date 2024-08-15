import torch
import torch.nn as nn


# Importing user made libraries
from dataset import *
from model import build_transformer
from utils import *
from config import *
from input_funcs import *

from pathlib import Path

import os


if __name__ == "__main__":

    # select device
    device = get_device()
    device = torch.device(device)

    # Get arguments from user
    getting_vars()

    ## Visually see what the variables are set to in the Command line interface
    ### DO NOT CHANGE THE VARIABLES IN HERE. CHANGE IT IN THE config.py file 
    print(f"Machine: {MACHINE}")
    print(f"Execution: {EXECUTION}")
    print(f"Input Splits: {INPUT_SPLITS}")
    print(f"Number of Heads: {NUM_HEADS}")
    print(f"Number of Encoder/Decoder Blocks: {NUM_ENC_DEC_BLOCKS}")
    print(f"Embeddings Dimension: {EMB_DIM}")
    print(f"Training Batch Size: {TRAINING_BATCH_SIZE}")
    print(f"Training Epochs: {TRAINING_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")


    # Data Paths
    current_dir = os.getcwd()

    ## Make sure the weights folder exists
    Path(f"{DATASOURCE}_{MODEL_FOLDER}").mkdir(parents=True, exist_ok=True)

    # Loading Data
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_ds(TRAINING_BATCH_SIZE)

    # Create Model
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size(), SEQ_LEN, SEQ_LEN, d_model=EMB_DIM,
                              N=NUM_ENC_DEC_BLOCKS, h=NUM_HEADS).to(device)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-9)

    if EXECUTION == 't':
        train_loop(model= model, loss_fn= loss_fn, optimizer= optimizer, train_dataloader=train_dataloader, 
                   val_dataloader=val_dataloader, tokenizer_src=tokenizer_src, tokenizer_trg=tokenizer_trg, device=device)

    if EXECUTION == 's':
        pass