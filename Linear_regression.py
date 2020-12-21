import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

x_np,y_np=datasets.make_regression(n_samples=100,n_features=1,noise=10,random_state=1)

x=torch.from_numpy(x_np.astype(np.float32))
y=torch.from_numpy(y_np.astype(np.float32))
y=y.view(y.shape[0],1)

num_samples,num_features=x.shape

class LinearRegression(nn.Module):
    def __init__(self,in_size,out_size):
        super(LinearRegression,self).__init__()
        self.lin=nn.Linear(in_size,out_size)
    
    def forward(self,x):
        return self.lin(x)

num_samples,num_features=x.shape

model=LinearRegression(num_features,num_features)


l_rate=0.1
num_epochs=100
loss=nn.MSELoss()
optimiser=torch.optim.SGD(model.parameters(),lr=l_rate)
for  epoch in range(num_epochs):
    #Forward prop
    y_pred=model(x)

    l=loss(y,y_pred)

    l.backward()

    optimiser.step()
    optimiser.zero_grad() 

y_pred=model(x).detach().numpy()
plt.plot(x_np,y_np,'ro')
plt.plot(x_np,y_pred,'b')
plt.show()