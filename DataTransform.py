import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self,transform=None):
        wine_data=np.loadtxt('./datasets/wine.csv',delimiter=',',skiprows=1,dtype=np.float32)
        self.x=wine_data[:,1:]
        self.y=wine_data[:,[0]]
        self.num_samples=wine_data.shape[0]
        self.transform=transform
    def __getitem__(self,ind):
        sample=self.x[ind],self.y[ind]
        if self.transform:
            sample=self.transform(sample)
        return sample
    def __len__(self):
        return self.num_samples


class toTensor:
    def __call__(self,sample):
        inputs,labels=sample
        return torch.from_numpy(inputs),torch.from_numpy(labels)

class mulTransform:
    def __init__(self,factor):
        self.factor=factor
    
    def __call__(self,sample):
        inputs,labels=sample
        inputs*=self.factor
        return inputs,labels


# dataset= WineDataset(transform=toTensor())
# features,labels=dataset[0]
# print(type(features),type(labels))

composed_transform=torchvision.transforms.Compose([toTensor(),mulTransform(4)])

dataset= WineDataset(transform=composed_transform)
features,labels=dataset[0]
print(type(features),type(labels))
print(features)