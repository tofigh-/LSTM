import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module, nonlinearity=None):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.nonlinearity = nonlinearity

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        time_length, batch_size = x.size(0), x.size(1)
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(time_length * batch_size, x.size(2))
        if self.nonlinearity is not None:
            y = self.nonlinearity(self.module(x_reshape))
        else:
            y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(time_length, batch_size, y.size()[1])
        return y
