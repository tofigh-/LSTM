import torch
from torch import nn

l1 = nn.Linear(3, 3)
l1.weight.data.fill_(0)
l1.bias.data.fill_(0)
x = (torch.ones(2, 3))
# backward one loss only
loss1 = 0
for i in range(1):
    y = l1(x)
    loss1 += (y + 1).abs().sum()
loss1.backward(retain_graph=True)
dd=[]
dd.append(l1.weight.grad)
l1.weight.grad = None
print dd
# # backward the other loss only
# loss2 = (y + 1).abs().sum()
# loss2.backward()
# print l1.weight.grad
