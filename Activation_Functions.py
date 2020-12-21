import numpy as np
import torch
import torch.nn as nn

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def cross_entropy(y,y_pred):
    loss=-np.sum(y*np.log(y_pred))
    return loss

x=np.array([1,2,-4])
y=softmax(x)
print(f'inputs->{x},outputs={y}')

x=torch.tensor([1.0,2.0,-4.0])
y=torch.softmax(x,dim=0)
print(f'inputs->{x},outputs={y}')

y=np.array([1,0,0])
y_pred=np.array([0.99,0.007,0.003])
print(cross_entropy(y,y_pred))

#Applies softmax also
loss=nn.CrossEntropyLoss()
y=torch.tensor([2,0,1])
y_pred=torch.tensor([[2.0,1.0,5.1],[2.0,1.0,0.1],[1.0,5.0,0.1]])
l=loss(y_pred,y)
print(l.item())