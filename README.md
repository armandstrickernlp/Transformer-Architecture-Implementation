# Transformer Architecture Implementation
This repo illustrates the inner workings of the transformer architecture, to get a better handle on its essential mechanisms and building blocks.
The code is in part from chapter 2 of Huggingface's transformers [book](https://www.amazon.fr/Natural-Language-Processing-Transformers-Applications/dp/1098103246/ref=sr_1_1?__mk_fr_FR=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=18DWPJDRYVTZ5&keywords=hugging+face&qid=1657359257&sprefix=huggingface%2Caps%2C61&sr=8-1)
which mainly focuses on the Encoder part of the full transfo architecture.  

When reading the code, you should follow the flow of the model : start with the Encoder, then move on to the Decoder 
and finally `Transformer.py` which brings it all together.  The model hyperparameters (num_layers, num_heads, embed_dim...) are based 
on the architecture of T5-small, an encoder-decoder type model.  The model's config has been printed in `config.txt` for reference.

