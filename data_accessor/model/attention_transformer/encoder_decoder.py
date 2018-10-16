import torch
from torch import nn
from data_accessor.data_loader.Settings import *
from noam_optimizer import NoamOpt
import numpy as np
from data_accessor.model.model_utilities import cuda_converter


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, embeddings, encoder, decoder, generator, model_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.embeddings = embeddings
        self.model_size = model_size
        self.optimizer = self.get_std_optimizer()
        self.loss_weights = {i: 1.0 * cuda_converter(nn.Parameter(torch.ones(1))) for i in range(NUM_COUNTRIES)}
        self.loss_weight_optimizer = torch.optim.Adam(self.loss_weights.values(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
        for p in self.loss_weights.values():
            p.requires_grad = False

    def forward(self, input_output_seq, input_mask, output_mask):
        "Take in and process masked src and target sequences."
        input_seq, output_seq = self.embed(input_output_seq)
        return self.decode(self.encode(input_seq, input_mask), input_mask, output_seq)

    def embed(self, input_output_seq):
        return self.embeddings(input_output_seq)

    def encode(self, input_seq, encoder_input_mask):
        return self.encoder(input_seq, encoder_input_mask)

    def decode(self, hidden_state, encoder_input_mask, decoder_input):
        return self.decoder(decoder_input, hidden_state, encoder_input_mask)

    def generate_mu_sigma(self, input):
        return self.generator(input)

    @staticmethod
    def load_checkpoint(model_path_dict, model):
        encoder_decoder_checkpoint = torch.load(model_path_dict[ENCODER_DECODER_CHECKPOINT])

        model.load_state_dict(encoder_decoder_checkpoint[STATE_DICT])

    @staticmethod
    def save_checkpoint(model, model_file_name):
        encoder_decoder_state = {
            STATE_DICT: model.state_dict(),
            OPTIMIZER: model.optimizer.state_dict()
        }

        torch.save(encoder_decoder_state, model_file_name)

    def mode(self, train_mode=True):
        if train_mode:
            self.train(True)
        else:
            self.eval()

    def get_std_optimizer(self):
        return NoamOpt(model_size=self.model_size, factor=2, warmup=10,
                       optimizer=torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9))


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return cuda_converter(torch.from_numpy(subsequent_mask) == 0)
