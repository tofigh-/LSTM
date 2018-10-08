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
from cnn_encoder import CNNEncoder


def make_model(embedding_descriptions, total_input, forecast_length, N=6,
               d_model=96, d_ff=4 * 96, h=8, dropout_enc=0.1, dropout_dec=0.1, num_cnn_filters=64,
               ngram_filter_sizes=[4, 8, 12, 16, 32]):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    multi_head_attn = MultiHeadedAttention(h, d_model)
    positionwise_feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_enc)
    embedding_sizes = [embedding_descriptions[feature][EMBEDDING_SIZE] for feature in embedded_features]
    num_embedded_features = sum(embedding_sizes)
    num_cnn_features = len(ngram_filter_sizes) * num_cnn_filters
    model = EncoderDecoder(
        embeddings=Embeddings(embedding_descriptions, total_input, forecast_length),
        cnn_encoder=CNNEncoder(
            embedding_dim=d_model,
            num_filters=num_cnn_filters,
            ngram_filter_sizes=ngram_filter_sizes),
        encoder=Encoder(EncoderLayer(d_model, c(multi_head_attn), c(positionwise_feed_forward), dropout_enc), N),
        decoder=Decoder(DecoderLayer(d_model, c(multi_head_attn), c(multi_head_attn),
                                     c(positionwise_feed_forward), dropout_dec), N),
        generator=GeneratorLayer(d_model * 2 + num_embedded_features + num_cnn_features, num_output=14),
        model_size=d_model)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
