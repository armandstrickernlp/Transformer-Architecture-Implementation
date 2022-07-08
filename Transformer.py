"""
The full Transformer architecture, putting together the Encoder and Decoder
as well as adding a language modeling head on top of the decoder.
This head is just a fully connected layer whose output size is the vocab size.
"""


from transformers import AutoTokenizer, AutoConfig
from torch import nn

import torch

from Encoder import TransformerEncoder
from Decoder import TransformerDecoder


class LanguageModelHead(nn.Module):
    """
    This head outputs logits for each sequence position.
    We can get a proba distrib for the next word by softmaxing
    the logits in the final position in the sequence.
    """
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fully_connected = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, dec_hidden_state):
        dec_hidden_state = self.dropout(dec_hidden_state)
        logits = self.fully_connected(dec_hidden_state)
        return logits


class Transformer(nn.Module):
    """
    Putting it all together. 
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.lm_head = LanguageModelHead(config)
    
    def forward(self, enc_input_ids, dec_input_ids):
        enc_out = self.encoder(enc_input_ids)
        dec_out = self.decoder(dec_input_ids, enc_out)
        out = self.lm_head(dec_out)
        return out


#############################################################################
def main():
    """
    Show the result of this untrained transformer architecture by passing in
    an encoder input and an unfinished decoder input.
    """
    torch.manual_seed(42)

    model_ckpt = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    config = AutoConfig.from_pretrained(model_ckpt) # config printed into a .txt file to look up each model hyperparameter
   
    # Encoder input
    text_in = "I love New York, I want to try and move there one day."
    inputs_enc = tokenizer(text_in, return_tensors='pt', add_special_tokens=False, padding=True) # padding if batch of examples

    # Decoder input
    text_out = "I want to live in"
    inputs_dec = tokenizer(text_out, return_tensors='pt', add_special_tokens=False, padding=True)
   

    # Output
    untrained_transformer = Transformer(config)
    logits = untrained_transformer(inputs_enc.input_ids, inputs_dec.input_ids)
    # get top 10 predicted next tokens
    top_10_next_tokens = tokenizer.convert_ids_to_tokens(torch.argsort(logits[:, -1, :], descending=True).squeeze(0)[:10])
    print(top_10_next_tokens)
    # just to see th output... since model is not trained, the next token is likely to be irrelevant 
    # with respect to the previous tokens

if __name__ == "__main__":
    main()