from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_trg, src_lang, trg_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_trg.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_trg.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_trg.token_to_id("[PAD]")], dtype=torch.int64)



    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
            src_target_pair = self.ds[index]
            src_text = src_target_pair['translation'][self.src_lang]
            trg_text = src_target_pair['translation'][self.trg_lang]

            enc_input_tokens = self.tokenizer_src.encode(src_text).ids
            dec_input_tokens = self.tokenizer_trg.encode(trg_text).ids
            
            enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2   # removing SOS and EOS token
            dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1   # removing SOS token only

            if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
                 raise ValueError('Sentence is too long')

            # source text
            encoder_input = torch.cat([
                 self.sos_token,
                 torch.tensor(enc_input_tokens, dtype=torch.int64),
                 self.eos_token,
                 torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ])

            # decoder input
            decoder_input = torch.cat([
                 self.sos_token,
                 torch.tensor(dec_input_tokens, dtype=torch.int64),
                 torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
                 # No end of sentence since we want more tokens to be generated
            ])

            # output of decoder (target sentence)
            label = torch.cat([
                 # Note there is no SOS token
                 torch.tensor(dec_input_tokens, dtype=torch.int64),
                 self.eos_token,
                 torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ])

            assert encoder_input.size(0) == self.seq_len
            assert decoder_input.size(0) == self.seq_len
            assert label.size(0) == self.seq_len

            return {
                 "encoder_input": encoder_input,
                 "decoder_input": decoder_input,
                 "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
                 # this line is ensuring that for all the words in a sentence the words are tokenized, the values for those positions
                 # is set to 1, while any padding tokens is set to 0 
                "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
                # The left of & is the same as the encoder_mask, but to the right of that we are making sure that the decoder does not
                # see tokens that it does not have access to in the target sentence
                "label": label,
                "src_text": src_text,
                "trg_text": trg_text
            }
    
def causal_mask(size):
    # This returns a mask such that all values above the diagonal of the matrix
    # is set to 1
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0   # since we want everything above the diagonal to be 0
    