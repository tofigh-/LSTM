import torch
import torch.tensor as T
def f(x, out):
    return x * out
x = torch.ones((2,2), requires_grad=True)
y=  2 * x
c = torch.ones((2,2),requires_grad = False)
for i in range(2):
    z = c * y * y * y
    # z.backward()
    # z.detach()
    u = c.clone()
    u[0,0] = z[0,0]
    c = u
    # c = z
    print "x req grad", x.requires_grad
# print z, z.requires_grad
z.backward(torch.FloatTensor([[1,1],[1,1]]))
print z
print x.grad

# import torch
# from torch.autograd import Variable
# x = torch.ones(3,requires_grad=True)
# z = x.sum()
# for i in range(1):
#     for j in range(3):
#         z = x.sum() * x.sum() + z
#         x[0] = z
#         print x,z
#     z.backward() # Calculate gradients
# print(x)
# print x.grad
