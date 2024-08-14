from pathlib import Path
import argparse

### Defaults for variables
seq_len = 350
datasource = 'opus_books'
lang_src = "en"
lang_trg = "it"
model_folder = "weights"
model_basename = "tmodel_"
PRELOAD = None
tokenizer_file = "tokenizer_{0}.json"

## Set if you're using your windows or mac machine (data directory location)
machine = 'w'     # m for mac, w for windows

## Set if setting the model in training mode or inference mode
execution = 't'     # t for train, s for sample

## Model Variables
input_splits = 3
num_heads = 8
num_enc_dec_blocks = 6
emb_dim = 512

## Hyperparameters
training_batch_size = 8
training_epochs = 50
learning_rate = 1e-3

def update_globals_from_args(args):
    global machine, execution, input_splits, num_heads, num_enc_dec_blocks, emb_dim
    global training_batch_size, training_epochs, learning_rate

    machine = args.machine
    execution = args.execution
    input_splits = args.input_splits
    num_heads = args.num_heads
    num_enc_dec_blocks = args.num_enc_dec_blocks
    emb_dim = args.emb_dim
    training_batch_size = args.training_batch_size
    training_epochs = args.training_epochs
    learning_rate = args.learning_rate

def getting_vars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', type=str, default=machine, choices=['w', 'm'])
    parser.add_argument('--execution', type=str, default=execution, choices=['t', 'i'], 
                        help='t for train, or i for inference the model')
    parser.add_argument('--input_splits', type=int, default=input_splits, 
                        help='# of splits of the input matrix')
    parser.add_argument('--num_heads', type=int, default=num_heads, 
                        help='# of heads in the transformer model')
    parser.add_argument('--num_enc_dec_blocks', type=int, default=num_enc_dec_blocks, 
                        help='# of encoder / decoder blocks to train')
    parser.add_argument('--emb_dim', type=int, default=emb_dim, 
                        help='size of the embeddings vector')
    parser.add_argument('--training_batch_size', type=int, default=training_batch_size, 
                        help='Batch Size during Training and Validation')
    parser.add_argument('--training_epochs', type=int, default=training_epochs, 
                        help='# of epochs you want to train the model')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, 
                        help='learning rate')

    args = parser.parse_args()
    
    update_globals_from_args(args)


# Example usage
if __name__ == "__main__":
    args = getting_vars()
    print(f"Machine: {machine}")
    print(f"Execution: {execution}")
    print(f"Input Splits: {input_splits}")
    print(f"Number of Heads: {num_heads}")
    print(f"Number of Encoder/Decoder Blocks: {num_enc_dec_blocks}")
    print(f"Embedding Dimension: {emb_dim}")
    print(f"Training Batch Size: {training_batch_size}")
    print(f"Training Epochs: {training_epochs}")
    print(f"Learning Rate: {learning_rate}")
