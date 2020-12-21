import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Loading Breast Cancer dataset
input_data=datasets.load_breast_cancer()

#splitiing the dataset into train and test
x,y=input_data.data,input_data.target

num_samples,num_features=x.shape

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)

#Normalising the dataset to zero mean and unit variance
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Changing numpy arrays to torch tensors
x_train=torch.from_numpy(x_train.astype(np.float32))
x_test=torch.from_numpy(x_test.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))

#Reshaping y tensors
y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)


class LogisticRegression(nn.Module):
    def __init__(self,in_size):
        super(LogisticRegression,self).__init__()
        self.lin=nn.Linear(in_size,1)
    
    def forward(self,x):
        y_pred=torch.sigmoid(self.lin(x))
        return y_pred


model=LogisticRegression(num_features)


l_rate=0.01
num_epochs=100
loss=nn.BCELoss()
optimiser=torch.optim.SGD(model.parameters(),lr=l_rate)
for  epoch in range(num_epochs):
    #Forward prop
    y_pred=model(x_train)

    l=loss(y_pred,y_train)

    l.backward()

    optimiser.step()
    optimiser.zero_grad() 

    if epoch%5==0:
        print(f'epoch:{epoch}/{num_epochs},curr-loss={l.item():.4f}')

with torch.no_grad():
    y_pred=model(x_test)
    y_pred_class=y_pred.round()
    accuracy=y_pred_class.eq(y_test).sum()/y_test.shape[0]
    print(f'accuracy={accuracy:.4f}')