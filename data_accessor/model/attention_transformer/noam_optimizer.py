import torch
from data_accessor.data_loader.Settings import *


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        # self._step += 1
        # rate = self.rate()
        # for p in self.optimizer.param_groups:
        #     p['lr'] = rate
        # self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            original_rate  = self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))
            default_lr = LEARNING_RATE
        return default_lr

    def state_dict(self):
        self.optimizer.state_dict()

    def load_state_dict(self):
        self.optimizer.load_state_dict()

    def zero_grad(self):
        self.optimizer.zero_grad()
