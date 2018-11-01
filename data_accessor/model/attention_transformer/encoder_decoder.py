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

    def __init__(self, embeddings,
                 encoder,
                 near_future_decoder,
                 far_future_decoder,
                 near_future_generator,
                 far_future_generator,
                 model_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder

        self.near_future_decoder = near_future_decoder
        self.far_future_decoder = far_future_decoder

        self.near_future_generator = near_future_generator
        self.far_future_generator = far_future_generator

        self.embeddings = embeddings
        self.model_size = model_size
        self.optimizer = self.get_std_optimizer()
        if torch.cuda.is_available():
            self.loss_weights = cuda_converter(torch.ones(len(list_l2_loss_countries + list_l1_loss_countries)))
        else:
            self.loss_weights = torch.ones(len(list_l2_loss_countries + list_l1_loss_countries))
        self.loss_weights.requires_grad = False

    def forward(self, input_output_seq, input_mask, output_mask):
        "Take in and process masked src and target sequences."
        input_seq, output_seq = self.embed(input_output_seq)
        return self.decode(self.encode(input_seq, input_mask), input_mask, output_seq)

    def embed(self, input_output_seq, mask_key=STOCK):
        return self.embeddings(input_output_seq, mask_key)

    def encode(self, input_seq, encoder_input_mask):
        return self.encoder(input_seq, encoder_input_mask)

    def decode(self, hidden_state, encoder_input_mask, decoder_input, is_near_future=True):
        if is_near_future:
            return self.near_future_decoder(decoder_input, hidden_state, encoder_input_mask)
        else:
            return self.far_future_decoder(decoder_input, hidden_state, encoder_input_mask)

    def generate_mu_sigma(self, input, is_near_future=True):
        if is_near_future:
            return self.near_future_generator(input)
        else:
            return self.far_future_generator(input)

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

    def mode(self, mode="predict"):
        if mode == "predict":
            self.eval()
        elif mode == "train_near_future":
            self.train(True)

            for module_name, module in self._modules.iteritems():
                if module_name == "far_future_decoder" or module_name == "far_future_generator":
                    for params in module.parameters():
                        params.requires_grad = False
                else:
                    for params in module.parameters():
                        params.requires_grad = True

        elif mode == "train_far_future":
            self.train(True)
            for module_name, module in self._modules.iteritems():
                if module_name == "far_future_decoder" or module_name == "far_future_generator":
                    for params in module.parameters():
                        params.requires_grad = True
                else:
                    for params in module.parameters():
                        params.requires_grad = False

        else:
            raise ValueError(
                "provided mode option is invalid. Valid options are predict, train_near_future and train_far_future")

    def get_std_optimizer(self):
        return NoamOpt(model_size=self.model_size, factor=2, warmup=10,
                       optimizer=torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9))


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return cuda_converter(torch.from_numpy(subsequent_mask) == 0)
