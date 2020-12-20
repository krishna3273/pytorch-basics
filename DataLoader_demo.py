# Dataset and DataLoader
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        wine_data=np.loadtxt('./datasets/wine.csv',delimiter=',',skiprows=1,dtype=np.float32)
        self.x=torch.from_numpy(wine_data[:,1:])
        self.y=torch.from_numpy(wine_data[:,[0]])
        self.num_samples=wine_data.shape[0]
    def __getitem__(self,ind):
        return self.x[ind],self.y[ind]
    def __len__(self):
        return self.num_samples


# Custom Dataset
dataset=WineDataset()

#Inbuilt Datasets
# dataset=torchvision.datasets.MNIST()
# print(len(dataset))

# features,labels=dataset[0]
# print(f"features={features} label={labels}")
# batch_size=5
# dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=4)

# data_iter=iter(dataloader)
# features,labels=data_iter.next()
# print(f"features={features} labels={labels}")

num_epochs=4
samples_size=len(dataset)
num_iters=math.ceil(samples_size/batch_size)

for curr_epoch in range(num_epochs):
    for i,(input_data,labels) in enumerate(dataloader):
        print(f"current epoch is {curr_epoch}/{num_epochs} and current step is {i}/{num_iters},shape of current batch size is {input_data.shape}")
