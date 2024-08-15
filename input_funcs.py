
# torch libraries
from torch.utils.data import Dataset, DataLoader, random_split

# Huggingface libraries
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
"""This is used to create a tokenizer that splits text based on a fixed vocabulary of words. 
It assigns a unique ID to each word in the vocabulary and uses these IDs for tokenization. 
This is useful for tasks where you want to work with word-level tokens rather than subword units."""

from tokenizers.trainers import WordLevelTrainer
"""This trainer is used to build and train a WordLevel tokenizer. 
It takes a list of sentences (or text) and creates a vocabulary based on the frequency 
of words in the text. The resulting vocabulary is then used by the WordLevel tokenizer 
to encode and decode text."""

from tokenizers.pre_tokenizers import Whitespace    # Splits the tokens by space
"""This is a pre-tokenizer that splits text into tokens based on whitespace. 
It’s a simple way to segment text into words or other basic units before applying 
further tokenization steps."""

from config import *
from dataset import *

from pathlib import Path


def get_all_sentences(ds, lang):
    ### Iterator to return a sentence from dataset for a 
    ### selected language

    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(ds, lang):
    ### Getting the tokens from Hugging Face, or building tokens 

    # If tokenizer_file is "tokenizer_{}.json" and lang is "en", then tokenizer_path would be "tokenizer_en.json"
    tokenizer_path = Path(TOKENIZER_FILE.format(lang))

    # If tokenizer file does not exist
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour

        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # Create tokenizer instance
        tokenizer.pre_tokenizer = Whitespace()              # Choose whitespace as a pretokenizer to split words based on white space
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        """Initializes a WordLevelTrainer with special tokens and a minimum frequency threshold. 
        Special tokens like [UNK] (unknown), [PAD] (padding), [SOS] (start of sequence), and [EOS] (end of sequence) 
        are added to the vocabulary, and a word must appear at least twice to be included in the vocabulary."""

        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer) # Trains tokenizer from the sentences in dataset
        tokenizer.save(str(tokenizer_path))     # Saves tokens to file

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))    # takes tokens from file

    return tokenizer



def get_ds(batch_size):
    ### Builds training and testing data corpus along with tokenization

    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(f"{DATASOURCE}", f"{LANG_SRC}-{LANG_TRG}", split='train')
    #print("**************************")
    #print(f"Length of Dataset: {len(ds_raw)}")
    #print(f"0.1% of the dataset: {ds_raw[0: int(0.01*len(ds_raw))]}")
    #print("**************************")
    #print("")

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(ds_raw, LANG_SRC)
    tokenizer_trg = get_or_build_tokenizer(ds_raw, LANG_TRG)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])    

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_trg, LANG_SRC, LANG_TRG, SEQ_LEN)
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_trg, LANG_SRC, LANG_TRG, SEQ_LEN)

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_trg = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][LANG_SRC]).ids
        trg_ids = tokenizer_trg.encode(item['translation'][LANG_TRG]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_trg = max(max_len_trg, len(trg_ids))
    
    """
    - item['translation'][lang_src]: Retrieves the source sentence for the current item from the dataset, 
    where lang_src is the language code for the source language (e.g., 'en' for English).

    - tokenizer_src.encode(...): Uses the source language tokenizer (tokenizer_src) to encode the source 
    sentence into token IDs.

    - .ids: Extracts the list of token IDs from the encoded output. The length of this list represents the 
    number of tokens (i.e., the length) of the encoded sentence."""

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_trg}')
    

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg