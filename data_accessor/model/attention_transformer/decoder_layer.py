import torch.nn as nn
from encoder import clones
from sublayer_connection import SublayerConnection


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, decoder_input, hidden_state, encoder_mask, decoder_mask):
        "Follow Figure 1 (right) for connections."
        m = hidden_state
        decoder_input = self.sublayer[0](decoder_input, lambda x: self.self_attn(x, x, x, decoder_mask))
        decoder_input = self.sublayer[1](decoder_input, lambda x: self.src_attn(x, m, m, encoder_mask))
        return self.sublayer[2](decoder_input, self.feed_forward)
