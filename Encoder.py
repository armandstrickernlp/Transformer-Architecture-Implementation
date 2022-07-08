"""
This script illustrates the **Encoder** architecture, the model config is taken from T5-small,
an encoder-decoder type transformer.
"""

from transformers import AutoTokenizer, AutoConfig
from torch import nn
from math import sqrt

import torch.nn.functional as F

import torch


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Implements self-attention, with an optional mask (for the decoder)
    Returns an updated vector representation for each token passing through the encoder
    Each new token representation is a "fancy average" of the other tokens' representations

    """
    dim_k = key.size(-1)
    scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float('inf')) # masks the attention scores of words to the right
                                                             # model can attend to previous tokens and present token only
    weights = F.softmax(scores, dim=-1) # square matrix of distributions (batch, seq_len, seq_len)
    
    return torch.bmm(weights, value) # bmm : batch matrix multi; output_dim (batch, seq_len, head_dim)


class AttentionHead(nn.Module):
    """
    1 attention head.
    The embedding of each token is projected into 3 spaces:
    -query
    -key
    -value
    These projections are fed to the self-attention mecanism

    The head_dim is a multiple of embed_dim.
    IN t5-small case here, head_dim = 512 / 8 = 64 
    See next class for head explanation.
    """
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim, bias=False)
        self.k = nn.Linear(embed_dim, head_dim, bias=False)
        self.v = nn.Linear(embed_dim, head_dim, bias=False)
    
    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state),
            self.k(hidden_state),
            self.v(hidden_state)
        )
        return attn_outputs


class MultiHeadAttention(nn.Module):
    """
    Embedding is projected into multiple query, key, value spaces so 
    several "types" of attention can be performed (semantic vs. syntactic 
    relations between tokens for ex.) These are known as attention heads.
    Outputs of each attention head are concatenated together and then projected again. 
    """
    def __init__(self, config):
        super().__init__()
        embed_dim = config.d_model
        num_heads = config.num_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, head_dim) for _ in range(num_heads) # [(B, seq_len, head_dim), (B, seq_len, head_dim)...]
        ])
        self.output_linear = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads],dim=-1)
        x = self.output_linear(x)
        return x



class FeedForward(nn.Module):
    """
    Position-wise feed forward network
    The whole sequence of embeddings is not processed as a single vector, 
    instead each token embedding is passed through the ffn independently.
    This is where most of the memorization is hypothesized to happen.
    First layer output size is typically *4 the embedding dim. (2048 in this case)
    """
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return self.dropout(x)



class TransformerEncoderBlock(nn.Module):
    """
    We add pre-layer normalization here (supposed to be more stable during training)
    + skip connections
    """
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.layer_norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
    
    def forward(self, hidden_state):
        x = self.layer_norm1(hidden_state) # normalize each input in the batch to have 0 mean and unity variance
        x = self.attention(x) + x # add the unprocessed embedding to the processed one
        x = self.layer_norm2(x)
        x = self.feed_forward(x) + x

        return x



class Embeddings(nn.Module):
    """
    Want to add positional information when doing our 'fancy weighted sum'...
    EAsy way to do so is add a trainable embedding layer, where instead of 
    each entry being a *token* index, the entry is a *position* index.
    Positional embeddings are then learned during training.
    The final embedded input to the encoder blocks is the sum
    of both embeddings(token_id + position_id).
    """
    def __init__(self, config):
        super().__init__()
        self.token_embedings = nn.Embedding(config.vocab_size,config.d_model)
        self.position_embeddings = nn.Embedding(config.n_positions,config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        seq_length = input_ids.size(1) 
        position_ids = torch.arange(seq_length, dtype=torch.long).expand(input_ids.size(0), seq_length) # dims : [B, seq_length]

        token_embeddings = self.token_embedings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class TransformerEncoder(nn.Module):
    """
    Putting it all together to form the encoder 
    """
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(config) for _ in range(config.num_layers)
        ])
    
    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x =layer(x)
        return x

#############################################################################
def main():
    """
    Input an example and show tensor size after passing through the encoder
    """
    model_ckpt = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    config = AutoConfig.from_pretrained(model_ckpt) # config printed into a .txt file to look up each model hyperparameter
   
    # encoder input
    text_in = "I love New York, I want to try and move there one day."
    inputs_enc = tokenizer(text_in, return_tensors='pt', add_special_tokens=False, padding=True) # if batch of examples
    print(f"Input size : {inputs_enc.input_ids.size()}")

    # encoder output
    encoder = TransformerEncoder(config)
    encoder_out = encoder(inputs_enc.input_ids)
    print(f"Encoder output size: {encoder_out.size()}")


if __name__ == "__main__":
    main()
    
