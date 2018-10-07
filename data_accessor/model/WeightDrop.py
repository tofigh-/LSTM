from torch.nn import Parameter

import torch


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    if isinstance(module, torch.nn.RNNBase): module.flatten_parameters = lambda *args, **kwargs: None
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))
        setattr(module, name_w, w.data)

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)

        return original_module_forward(*args)

    setattr(module, 'forward', forward)


class WeightDrop(torch.nn.Module):
    """
    The weight-dropped module applies recurrent regularization through a DropConnect mask on the
    hidden-to-hidden recurrent weights.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        module (:class:`torch.nn.Module`): Containing module.
        weights (:class:`list` of :class:`str`): Names of the module weight parameters to apply a
          dropout too.
        dropout (float): The probability a weight will be dropped.

    Example:

        from torchnlp.nn import WeightDrop
         import torch

        gru = torch.nn.GRUCell(2, 2)
        weights = ['weight_hh']
        weight_drop_gru = WeightDrop(gru, weights, dropout=0.9)

        input_ = torch.randn(3, 2)
        hidden_state = torch.randn(3, 2)
        weight_drop_gru(input_, hidden_state)
        -0.4467 -0.1344
        0.1747  0.9075
        0.2340  0.1977
        [torch.FloatTensor of size 3x2]
    """

    def __init__(self, module, weights, dropout=0.0):
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, dropout)
        self.module = module
        self.forward = module.forward


class WeightDropLSTM(torch.nn.LSTM):
    """
    Wrapper around :class:`torch.nn.LSTM` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, weight_dropout=0.0, *args, **kwargs):
        super(WeightDropLSTM).__init__()
        torch.nn.LSTM.__init__(self, *args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


class WeightDropGRU(torch.nn.GRU):
    """
    Wrapper around :class:`torch.nn.GRU` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, weight_dropout=0.0, *args, **kwargs):
        super(WeightDropGRU).__init__(*args, **kwargs)
        torch.nn.GRU.__init__(self, *args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


class WeightDropLinear(torch.nn.Linear):
    """
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, weight_dropout=0.0, *args, **kwargs):
        super(WeightDropLinear).__init__(*args, **kwargs)
        torch.nn.Linear.__init__(self, *args, **kwargs)
        weights = ['weight']
        _weight_drop(self, weights, weight_dropout)
