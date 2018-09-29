import copy
from attention import MultiHeadedAttention
from positionwise_feedforward import PositionwiseFeedForward
from encoder_decoder import EncoderDecoder
from encoder import Encoder
from encoder_layer import EncoderLayer
from decoder import Decoder
from decoder_layer import DecoderLayer
from torch import nn
from embeddings import Embeddings
from generator_layer import GeneratorLayer


def make_model(embedding_descriptions, total_input, forecast_length, N=6,
               d_model=512, d_ff=2048, h=8, dropout_enc=0.1, dropout_dec=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout_enc)

    model = EncoderDecoder(
        embeddings=Embeddings(embedding_descriptions, total_input, forecast_length),
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout_enc), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                     c(ff), dropout_dec), N),
        generator=GeneratorLayer(d_model, num_output=14))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
