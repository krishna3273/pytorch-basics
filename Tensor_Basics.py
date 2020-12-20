import torch
import numpy as np
# x=torch.empty(2,3)
# x=torch.rand(2,3)
# x=torch.zeros(2,3)
# x=torch.ones(2,2,dtype=torch.int64)
# x=torch.tensor([1,3,4])
# print(x,x.dtype)


x1=torch.rand(3,3)
x2=torch.rand(3,3)
# print(x1)
# print(x2)
#Normal Addition
# print(x1+x2,torch.add(x1,x2))
#Inplace Addition
# x2.add_(x1)

x=torch.rand(4,4)
# print(x)
#Slicing
# print(x[:,[0]])
#Reshaping
# x_temp=x.view(16)
# x_temp=x.view(-1,8)
# print(x_temp)


#Torch tensors to numpy arrays
x=torch.ones(5)
# print(x,type(x),x.size())
x=x.numpy()
# print(x,type(x),x.shape)

#numpy arrays to torch tensors
x=np.ones(5)
# print(x,type(x),x.shape)
x=torch.from_numpy(x)
# print(x,type(x),x.size())

#Tells that gradient of x needs to be calculated later
x=torch.ones(5,requires_grad=True)
# print(x)

