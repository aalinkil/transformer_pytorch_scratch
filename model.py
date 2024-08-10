import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

        def forward(self, x):
            return self.embedding(x) * math.sqrt(self.d_model) # shown in Attention paper
        
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)


        # Creating a zero matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        ## positional encoding formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueese(0)    # (1, seq_len, d_model) since we will have a batch of sentences

        # We register the tensor to the buffer of the model
        # This way the tensor will be saved when saving the model to a file
        self.register_buffer('pe', pe)

    def forward(self, x):

        # This is adding the positional encoding to x, and making sure that
        # this will not be a learned parameter (hence require_grad is false)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)  # dropout layer
        

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))    # nn.Parameter makes a learnable parameter
        self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            mean = x.mean(dim = -1, keepdim = True)
            std = x.std(dim = -1, keepdim = True)
            return self.alpha * (x - mean) / (std + self.eps) + self.bias
        

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)    # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)    # W2 and B2

    def forward(self, x):

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_len, d_model)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x 


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        # verify that you can split the embeddings into perfectly divisible heads
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model, d_model)  # Wq Bq
        self.w_k = nn.Linear(d_model, d_model)  # Wk Bk
        self.w_v = nn.Linear(d_model, d_model)  # Wv Bv
        self.w_o = nn.Linear(d_model, d_model)  # Wo Bo
        self.dropout = nn.Dropout(dropout)



    # Allows you to call this method without having a MultiHeadAttentionBlock object created
    # You can do it by just using `MultiHeadAttentionBlock.attention()
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        ### The attention calculations take place here
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)

        d_k = query.shape[-1]

        # @ in pytorch is for matrix multiplication (dot product of each row and column of two matrixes)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # if there is a mask defined we want to set the values for it to -ve infinity
        # mask is going to be a tensor, so you're saying in all locations where mask is 0
        # set the value to -ve inifinity
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Applying softmax after masks (if specified)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len)

        # applying dropout
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # multiply attention_scores to value
        return (attention_scores @ value), attention_scores # will be using attention_scores for visualization


    def forward (self, q, k, v, mask):

        query = self.w_q(q)     # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)       # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)     # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        ## divide each of the matrixes into different heads
        # view returns the same data but with a different shape
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)    # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k)
        # Transpose swaps the 2nd and 3rd dimension (index of columns starts from 0)
        query = query.transpose(1, 2)   # (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)

        # Now we do the same transformations for the key and the value matrixes (but in one line)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  # contiguous is used to put entire tensor
                                                                                    # in one block of memory
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    

class ResidualConnectionBlock(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    
    def forward(self, x, sublayer):
        ## This does the skipped connection and the Add & Norm block
        # I am not sure what x is and what sublayer is
        # Possible that the sublayer is the multi-headed-attention block, and x is the original input
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        
        # Processing the first residual connection
        attention_out = self.self_attention_block(x, x, x, src_mask)    # multi head attn output

        """attention_out vs. lambda x: attention_out:

            - attention_out is the result of the attention operation, which is a tensor.
            - lambda x: attention_out is a function that, when called with any input x, 
            will return the precomputed attention_out."""
        
        x = self.residual_connections[0](x, lambda x: attention_out)    # doing skip connection original input x, and the function attention_out
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(3)])

    
    def forward(self, x, encoder_output, src_mask, trg_mask):

        # src_mask is coming from the encoder (original language)
        # trg_mask is coming from the decoder (target language)

        attention_out = self.self_attention_block(x,x,x, trg_mask)
        x = self.residual_connections[0](x, lambda x: attention_out)
        attention_out = self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        x = self.residual_connections[1](x, lambda x: attention_out)
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.porj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        # log_softmax is applied for numerical stability
        return torch.log_softmax(self.proj(x), dim = -1)
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, trg_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, trg_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.projection_layer = projection_layer


    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    

    def decode(self, encoder_output, src_mask, trg, trg_mask):
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    


def build_transformer(src_vocab_size: int, trg_vocab_size: int, src_seq_len: int, trg_seq_len: int, 
                      d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> None:

    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    trg_embed = InputEmbeddings(d_model, trg_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)


    # create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, trg_vocab_size)

    # Create the transformer
    transformer  = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection_layer)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() >1:
            nn.init.xavier_uniform_(p)

    return transformer