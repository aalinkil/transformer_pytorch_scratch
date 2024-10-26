from pathlib import Path
import argparse

### Defaults for variables
#*************************************************************************************#
SEQ_LEN = 790
DATASOURCE = 'opus_books'
LANG_SRC = "en"
LANG_TRG = "es"
MODEL_FOLDER = "weights"
MODEL_BASENAME = "tmodel_"
PRELOAD = "latest"
TOKENIZER_FILE = "tokenizer_{0}.json"

## Set if you're using your windows or mac machine (data directory location)
MACHINE = 'w'     # m for mac, w for windows

## Set if setting the model in training mode or inference mode
EXECUTION = 't'     # t for train, s for sample

## Model Variables
INPUT_SPLITS = 3
NUM_HEADS = 8
NUM_ENC_DEC_BLOCKS = 6
EMB_DIM = 64

## Hyperparameters
TRAINING_BATCH_SIZE = 8
TRAINING_EPOCHS = 30
LEARNING_RATE = 1e-3

## For Tensorboard Graph
EXPERIMENT_NAME = "smallx100_test"

#*************************************************************************************#

def update_globals_from_args(args):
    global MACHINE, EXECUTION, INPUT_SPLITS, NUM_HEADS, NUM_ENC_DEC_BLOCKS, EMB_DIM
    global TRAINING_BATCH_SIZE, TRAINING_EPOCHS, LEARNING_RATE

    MACHINE = args.machine
    EXECUTION = args.execution
    INPUT_SPLITS = args.input_splits
    NUM_HEADS = args.num_heads
    NUM_ENC_DEC_BLOCKS = args.num_enc_dec_blocks
    EMB_DIM = args.emb_dim
    TRAINING_BATCH_SIZE = args.training_batch_size
    TRAINING_EPOCHS = args.training_epochs
    LEARNING_RATE = args.learning_rate

def getting_vars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default=MACHINE, choices=['w', 'm'])
    parser.add_argument('--execution', type=str, default=EXECUTION, choices=['t', 'i'], 
                        help='t for train, or i for inference the model')
    parser.add_argument('--input_splits', type=int, default=INPUT_SPLITS, 
                        help='# of splits of the input matrix')
    parser.add_argument('--num_heads', type=int, default=NUM_HEADS, 
                        help='# of heads in the transformer model')
    parser.add_argument('--num_enc_dec_blocks', type=int, default=NUM_ENC_DEC_BLOCKS, 
                        help='# of encoder / decoder blocks to train')
    parser.add_argument('--emb_dim', type=int, default=EMB_DIM, 
                        help='size of the embeddings vector')
    parser.add_argument('--training_batch_size', type=int, default=TRAINING_BATCH_SIZE, 
                        help='Batch Size during Training and Validation')
    parser.add_argument('--training_epochs', type=int, default=TRAINING_EPOCHS, 
                        help='# of epochs you want to train the model')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, 
                        help='learning rate')

    args = parser.parse_args()
    
    update_globals_from_args(args)


# Example usage
if __name__ == "__main__":
    args = getting_vars()
    print(f"Machine: {MACHINE}")
    print(f"Execution: {EXECUTION}")
    print(f"Input Splits: {INPUT_SPLITS}")
    print(f"Number of Heads: {NUM_HEADS}")
    print(f"Number of Encoder/Decoder Blocks: {NUM_ENC_DEC_BLOCKS}")
    print(f"Embedding Dimension: {EMB_DIM}")
    print(f"Training Batch Size: {TRAINING_BATCH_SIZE}")
    print(f"Training Epochs: {TRAINING_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
