import copy
from multi_head_attention import MultiHeadedAttention
from positionwise_feedforward import PositionwiseFeedForward
from encoder_decoder import EncoderDecoder
from encoder import Encoder
from encoder_layer import EncoderLayer
from decoder import Decoder
from decoder_layer import DecoderLayer
from torch import nn
from embeddings import Embeddings
from generator_layer import GeneratorLayer

from data_accessor.data_loader.Settings import *


def make_model(embedding_descriptions, total_input, forecast_length, N_enc=6,N_dec=6,
               d_model=96, d_ff=4 * 96, h=8, dropout_enc=0.1, dropout_dec=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    multi_head_attn = MultiHeadedAttention(h, d_model)
    positionwise_feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_enc)
    num_embedded_features = EMBEDDING_SIZE_FOR_ALL

    model = EncoderDecoder(
        embeddings=(Embeddings(embedding_descriptions, total_input, forecast_length)),
        encoder=(Encoder(EncoderLayer(d_model, c(multi_head_attn), c(positionwise_feed_forward), dropout_enc), N_enc)),
        decoder=(Decoder(DecoderLayer(d_model, c(multi_head_attn), c(multi_head_attn),
                                     c(positionwise_feed_forward), dropout_dec), N_dec)),
        generator=(GeneratorLayer(d_model * 2 + num_embedded_features, num_output=14)),
        model_size=d_model)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.orthogonal_(p)
    return model
