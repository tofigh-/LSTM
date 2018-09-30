import torch.nn as nn
import copy
from torch.nn import LayerNorm
from encoder import clones


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, encoder_input_mask, decoder_input_mask):
        for layer in self.layers:
            x = layer(x, memory, encoder_input_mask, decoder_input_mask)
        return self.norm(x)
