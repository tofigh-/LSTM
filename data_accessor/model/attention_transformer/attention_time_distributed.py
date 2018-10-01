from torch import nn


class TimeDistributed(nn.Module):
    def __init__(self, module, nonlinearity=None):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.nonlinearity = nonlinearity

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        batch_size, num_heads, time_length, num_features = x.shape
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(time_length * batch_size * num_heads, num_features)
        if self.nonlinearity is not None:
            y = self.nonlinearity(self.module(x_reshape))
        else:
            y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(batch_size, num_heads, time_length, y.shape[1]).transpose(-2,-1)
        return y
