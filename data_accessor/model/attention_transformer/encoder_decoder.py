import torch
from torch import nn
from data_accessor.data_loader.Settings import *
from noam_optimizer import NoamOpt

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

    def forward(self, input_output_seq, input_mask, output_mask):
        "Take in and process masked src and target sequences."
        input_seq, output_seq = self.embed(input_output_seq)
        return self.decode(self.encode(input_seq, input_mask), input_mask,
                           output_seq, output_mask)

    def embed(self, input_output_seq):
        return self.embeddings(input_output_seq)

    def encode(self, input_seq, src_mask):
        return self.encoder(self.src_embed(input_seq), src_mask)

    def decode(self, hidden_state, input_mask, output_seq, output_mask):
        return self.decoder(output_seq, hidden_state, input_mask, output_mask)

    @staticmethod
    def load_checkpoint(model_path_dict, model, model_optimizer):
        encoder_decoder_checkpoint = torch.load(model_path_dict[ENCODER_DECODER_CHECKPOINT])

        model.load_state_dict(encoder_decoder_checkpoint[STATE_DICT])
        model_optimizer.load_state_dict(encoder_decoder_checkpoint[OPTIMIZER])

    @staticmethod
    def save_checkpoint(model, optimizer, model_file_name):
        encoder_decoder_state = {
            STATE_DICT: model.state_dict(),
            OPTIMIZER: optimizer.state_dict()
        }

        torch.save(encoder_decoder_state, model_file_name)

    @staticmethod
    def mode(model, train_mode=True):
        if train_mode:
            model.train(True)
        else:
            model.eval()

    def get_std_optimizer(self):
        return NoamOpt(model_size=self.model_size, factor=2, warmup=4000,
                       optimizer=torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
