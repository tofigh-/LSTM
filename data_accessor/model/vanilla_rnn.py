import random

import torch
from torch import optim

from EncoderRNN import EncoderRNN
from FutureDecoder import FutureDecoder
from FutureDecoderWithAttention import FutureDecoderWithAttention
from data_accessor.data_loader.Settings import *
from model_utilities import cuda_converter,exponential


class VanillaRNNModel(object):

    def __init__(self, embedding_descripts, load_saved_model=True, is_attention=True, model_path_dict=None,
                 num_output=1):
        if load_saved_model and model_path_dict is None:
            raise ValueError('If load_saved_model is True, model_path_dict should be provided')
        self.encoder = cuda_converter(EncoderRNN(hidden_size=HIDDEN_SIZE,
                                                 embedding_descriptions=embedding_descripts,
                                                 n_layers=NUM_LAYER,
                                                 rnn_dropout=RNN_DROPOUT,
                                                 bidirectional=BI_DIRECTIONAL))
        if is_attention:
            self.future_decoder = cuda_converter(FutureDecoderWithAttention(self.encoder.batch_norm,
                                                                            embedding_descripts,
                                                                            n_layers=1,
                                                                            rnn_layer=self.encoder.rnn,
                                                                            num_output=num_output))
        else:
            self.future_decoder = cuda_converter(FutureDecoder(self.encoder.batch_norm,
                                                               embedding_descripts,
                                                               n_layers=1,
                                                               rnn_layer=self.encoder.rnn,
                                                               num_output=num_output))
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=LEARNING_RATE)
        self.future_decoder_optimizer = optim.Adam(self.future_decoder.parameters(), lr=LEARNING_RATE)

        if load_saved_model:
            self.load_checkpoint(model_path_dict)
        self.sales_col = feature_indices[SALES_MATRIX]

        # Alias predict with decode_output
        self.predict = self.decode_output

    def load_checkpoint(self, model_path_dict):
        encoder_checkpoint = torch.load(model_path_dict[ENCODER_CHECKPOINT])
        future_decoder_checkpoint = torch.load(model_path_dict[FUTURE_DECODER_CHECKPOINT])

        self.encoder.load_state_dict(encoder_checkpoint[STATE_DICT])
        self.encoder_optimizer.load_state_dict(encoder_checkpoint[OPTIMIZER])

        self.future_decoder.load_state_dict(future_decoder_checkpoint[STATE_DICT])
        self.future_decoder_optimizer.load_state_dict(future_decoder_checkpoint[OPTIMIZER])

    def save_checkpoint(self, encoder_file_name, future_decoder_file_name):
        encoder_state = {
            STATE_DICT: self.encoder.state_dict(),
            OPTIMIZER: self.encoder_optimizer.state_dict()
        }
        future_decoder_state = {
            STATE_DICT: self.future_decoder.state_dict(),
            OPTIMIZER: self.future_decoder_optimizer.state_dict()
        }
        torch.save(encoder_state, encoder_file_name)
        torch.save(future_decoder_state, future_decoder_file_name)

    def mode(self, train_mode=True):
        if train_mode:
            self.encoder.train(True), self.future_decoder.train(True)
        else:
            self.encoder.eval(), self.future_decoder.eval()

    def train(self, inputs, targets_future, loss_function, loss_function2, teacher_forcing_ratio):

        sales_future = targets_future[SALES_MATRIX]  # OUTPUT_SIZE x BATCH x NUM_COUNTRIES
        global_sales = targets_future[GLOBAL_SALE]
        # Teacher forcing: Feed the target as the next input while not teacher forcing consume the last prediciton as the input
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        loss = 0
        hidden_state = None
        embedded_features = None
        future_unknown_estimates = None
        all_week_predictions = []
        global_sale_all_weeks = []
        for future_week_idx in range(OUTPUT_SIZE):
            output_global_sale, \
            out_sales_predictions, \
            hidden_state, \
            embedded_features = self.decode_output(inputs,
                                                   future_week_idx,
                                                   hidden_state,
                                                   embedded_features,
                                                   future_unknown_estimates,
                                                   train=True
                                                   )
            all_week_predictions.append(out_sales_predictions)
            global_sale_all_weeks.append(output_global_sale)
            if out_sales_predictions.shape[0] == 0:
                print ("output_prediction is empty", out_sales_predictions)
                print inputs
                print hidden_state
                raise Exception

            loss += loss_function(exponential(out_sales_predictions[:, 1:],LOG_TRANSFORM),
                                  exponential(sales_future[future_week_idx, :, 1:],LOG_TRANSFORM))
            loss += loss_function2(exponential(out_sales_predictions[:, 0],LOG_TRANSFORM),
                                   exponential(sales_future[future_week_idx, :, 0],LOG_TRANSFORM))
            loss += loss_function(exponential(output_global_sale,LOG_TRANSFORM),
                                  exponential(global_sales[future_week_idx, :],LOG_TRANSFORM))
            if use_teacher_forcing:
                future_unknown_estimates = sales_future.data[future_week_idx, :, :]
            else:
                # without teacher forcing
                future_unknown_estimates = out_sales_predictions

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), GRADIENT_CLIP)
        torch.nn.utils.clip_grad_norm_(self.future_decoder.parameters(), GRADIENT_CLIP)

        self.encoder_optimizer.step()
        self.encoder_optimizer.zero_grad()
        self.future_decoder_optimizer.step()
        self.future_decoder_optimizer.zero_grad()

        return loss.item() / (OUTPUT_SIZE), global_sale_all_weeks, all_week_predictions

    def encode_input(self, inputs):
        input_seq = inputs[0]  # PAST_KNOWN_LENGTH * BATCH * TOTAL_NUM_FEAT
        batch_size = input_seq.shape[1]
        encoder_hidden = self.encoder.initHidden(batch_size)
        # encoder_hidden: num_layer x BATCH x HIDDEN_SIZE
        encoder_output, encoder_hidden, embedded_features = self.encoder(input_seq, encoder_hidden)
        return encoder_hidden, embedded_features, encoder_output

    def decode_output(self,
                      inputs,
                      future_week_index,
                      hidden_state=None,
                      embedded_features=None,
                      future_unknown_estimates=None,
                      train=False
                      ):
        '''
        :param inputs: # (TOTAL_INPUT x BATCH x TOTAL_FEAT, OUTPUT_SIZE * BATCH * TOTAL_FEAT)
        :param hidden_state:
        :param embedded_features: list of embedded features  len = num of embedded features, size of each element: BATCH_SIZE x Embedding_Size
        :param future_unknown_estimates: the sales of the last week used as value for sale
        :param future_week_index: start from 0
        :return:
        '''

        input_seq_decoder = inputs[1]
        encoder_outputs = None
        if hidden_state is None:
            hidden_state, embedded_features, encoder_outputs = self.encode_input(inputs)
        if future_unknown_estimates is None:
            input_seq_decoder[future_week_index, :, self.sales_col] = inputs[0].data[TOTAL_INPUT - 1, :,
                                                                      self.sales_col]
        else:
            if train:
                # To make sure gradients flow over time
                temp_input = input_seq_decoder.clone()
                temp_input[future_week_index, :, self.sales_col] = future_unknown_estimates
                input_seq_decoder = temp_input
            else:
                input_seq_decoder[future_week_index, :, self.sales_col].data = future_unknown_estimates

        future_decoder_hidden = hidden_state
        output_global_sale, out_sales_predictions, hidden = self.future_decoder(
            input=input_seq_decoder[future_week_index, :, :],
            hidden=future_decoder_hidden,
            embedded_inputs=embedded_features,
            encoder_outputs=encoder_outputs)
        return output_global_sale, out_sales_predictions, hidden, embedded_features

    def predict_over_period(self, inputs,
                            hidden_state=None,
                            embedded_features=None,
                            future_unknown_estimates=None,
                            ):
        all_week_predictions = []
        global_sale_all_weeks = []
        for week_idx in range(OUTPUT_SIZE):
            global_sales_prediction, future_unknown_estimates, hidden_state, embedded_features = self.decode_output(
                inputs,
                week_idx,
                hidden_state,
                embedded_features,
                future_unknown_estimates,
                train=False
            )
            all_week_predictions.append(future_unknown_estimates)
            global_sale_all_weeks.append(global_sales_prediction)
        return global_sale_all_weeks, all_week_predictions
