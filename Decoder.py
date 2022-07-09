"""
Components specific to the decoder have been re-coded, following the same
structure as in the encoder classes, making explicit the slight variations
appropriate for the decoder block.  The other classes/functions are imported from the Encoder.
"""

from transformers import AutoTokenizer, AutoConfig
from torch import nn

import torch

from Encoder import scaled_dot_product_attention, FeedForward, Embeddings, TransformerEncoder



class MaskedAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        """
        Same as in encoder except mask is added in the forward.
        Masked scores in the attention function:
        [score, -inf, -inf, ...]
        [score, score, -inf, ...]
        [score, score, score, -inf ...]
        ...
        """
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim, bias=False)
        self.k = nn.Linear(embed_dim, head_dim, bias=False)
        self.v = nn.Linear(embed_dim, head_dim, bias=False)
    
    def forward(self, hidden_state):
        seq_len = hidden_state.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len)).expand(hidden_state.size(0), seq_len, seq_len)
        #print(mask.size())
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state),
            self.k(hidden_state),
            self.v(hidden_state), 
            mask=mask # mask is not None here
        )
        return attn_outputs


class MaskedMultiHeadAttention(nn.Module):
    """
    Same as in the encoder except it uses the MaskedAttentionHead class above
    """
    def __init__(self, config):
        super().__init__()
        embed_dim = config.d_model
        num_heads = config.num_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([
            MaskedAttentionHead(embed_dim, head_dim) for _ in range(num_heads)
        ])
        self.output_linear = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads],dim=-1)
        x = self.output_linear(x)
        return x


class EncoderDecoderAttentionHead(nn.Module):
    """
    The MultiHeadAttention component in the decoder
    uses the outputs of the encoder and projects them into the 
    key and value spaces.  The decoder embeddings
    passed from the masked multi head attention module 
    are projected into the query space.
    This class encodes a single head.
    """
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim, bias=False)
        self.k = nn.Linear(embed_dim, head_dim, bias=False)
        self.v = nn.Linear(embed_dim, head_dim, bias=False)
    
    def forward(self, dec_hidden_state, enc_hidden_state): # add enc_hidden_state to the input
        attn_outputs = scaled_dot_product_attention(
            self.q(dec_hidden_state),
            self.k(enc_hidden_state), # k from encoder
            self.v(enc_hidden_state) # v from encoder
        )
        return attn_outputs


class MultiHeadEncoderDecoderAttention(nn.Module):
    """
    Sames as for encoder MuliheadAttention except we need to pass the encoder output
    in the forward.
    """
    def __init__(self, config):
        super().__init__()
        embed_dim = config.d_model
        num_heads = config.num_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([
            EncoderDecoderAttentionHead(embed_dim, head_dim) for _ in range(num_heads)
        ])
        self.output_linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, dec_hidden_state, enc_hidden_state):
        x = torch.cat([h(dec_hidden_state, enc_hidden_state) for h in self.heads],dim=-1)
        x = self.output_linear(x)
        return x


class TransformerDecoderBlock(nn.Module):
    """
    Full decoder block with layer normalization and skip connections.
    There are 3 main components:
    -Masked multihead attention
    -multihead encoder decoder attention
    -position-wise feed forward net
    """
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.layer_norm3 = nn.LayerNorm(config.d_model)
        self.masked_multihead_attention = MaskedMultiHeadAttention(config)
        self.multihead_encoder_decoder_attention = MultiHeadEncoderDecoderAttention(config)
        self.feed_forward = FeedForward(config)
    
    def forward(self, dec_hidden_state, enc_hidden_state):
        
        # normalize, masked attention + residual connection
        dec_hidden_state = self.layer_norm1(dec_hidden_state)
        dec_hidden_state = self.masked_multihead_attention(dec_hidden_state) + dec_hidden_state


        # normalize, encoder decoder attention + residual
        dec_hidden_state = self.layer_norm2(dec_hidden_state)
        dec_hidden_state = self.multihead_encoder_decoder_attention(dec_hidden_state, enc_hidden_state) + dec_hidden_state


        # normailze, feed-forward net + residual
        dec_hidden_state = self.layer_norm3(dec_hidden_state)
        dec_hidden_state = self.feed_forward(dec_hidden_state) + dec_hidden_state

        return dec_hidden_state


class TransformerDecoder(nn.Module):
    """
    Full Decoder 
    """
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(config) for _ in range(config.num_decoder_layers)
        ])
    
    def forward(self, input_ids, encoded_out):
        dec_hidden_state = self.embeddings(input_ids)
        for layer in self.layers:
            dec_hidden_state =layer(dec_hidden_state, encoded_out)
        return dec_hidden_state



#############################################################################

def main():
    """
    Input an example and show tensor size after passing through the encoder and decoder.
    For the encoder decoder attention to work, we need the encoder output as
    well as the decoder input.
    """
    model_ckpt = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    config = AutoConfig.from_pretrained(model_ckpt) # config printed into a .txt file to look up each model hyperparameter
   
    ## Encoder
    # encoder input
    text_in = "I love New York, I want to try and move there one day."
    inputs_enc = tokenizer(text_in, return_tensors='pt', add_special_tokens=False, padding=True) # if batch of examples
    print(f"Encoder Input size : {inputs_enc.input_ids.size()}")
    # encoded output
    encoder = TransformerEncoder(config)
    encoded_out = encoder(inputs_enc.input_ids)
    print(f"Encoder Output size : {encoded_out.size()}")
    
    ## Decoder
    # decoder input
    text_out = "I want to live in"
    inputs_dec = tokenizer(text_out, return_tensors='pt', add_special_tokens=False, padding=True)
    print(f"Decoder Input size : {inputs_dec.input_ids.size()}")
    # embed
    decoder = TransformerDecoder(config)
    decoder_out = decoder(inputs_dec.input_ids, encoded_out) # pass in input ids + encoder output 
    print(f"Decoder Output size: {decoder_out.size()}")


if __name__ == "__main__":
    main()
    