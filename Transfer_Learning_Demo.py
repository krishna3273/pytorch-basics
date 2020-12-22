import os
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets,transforms,models
import torchvision
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data_dir = './datasets/hymenoptera_data'
# image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),transform=transforms.ToTensor())
# train_dataloader =torch.utils.data.DataLoader(image_dataset, batch_size=1)
# data = iter(train_dataloader)
# features,labels=data.next()
# print(features.size())
# print(features.mean(), features.std())

#Calculated from above commented code(not necessary to use this transform)
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

#Helper function for plotting images
def imshow(inp, title):
    #plt needs channels as lastdim so doing transpose
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

#Loading dataset and creating batches
data_dir = './datasets/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,shuffle=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
# print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Demo plots of training data
# inputs, classes = next(iter(dataloaders['train']))
# #nrow defaults to 8 if no value is mentioned
# out = torchvision.utils.make_grid(inputs,nrow=2)
# imshow(out, title=[class_names[x] for x in classes])

#train_model(Neural Net Model,loss criterion,opt,learning rate scheduler,epochs)
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    #Storing starting time
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#Option-1(Train entire model again)
# model=models.resnet18(pretrained=True)
# num_features=model.fc.in_features
# #Changing last fully connected layer of resnet to suit our problem
# model.fc=nn.Linear(num_features,2)
# model.to(device)
# loss=nn.CrossEntropyLoss()
# optimiser=optim.SGD(model.parameters(),lr=0.01)
# #scheduler for updating learning rate
# #Every 7 epochs learning rate reduced to 10% of curr
# step_lr_scheduler=lr_scheduler.StepLR(optimizer=optimiser,step_size=7,gamma=0.1)
# model=train_model(model,loss,optimiser,scheduler=step_lr_scheduler,num_epochs=10)

#Option-2(Train only last part added newly)
model=models.resnet18(pretrained=True)
#Freezing all the already learnt parameters
for param in model.parameters():
    param.requires_grad=False
num_features=model.fc.in_features

#Changing last fully connected layer of resnet to suit our problem
model.fc=nn.Linear(num_features,2)
model.to(device)

loss=nn.CrossEntropyLoss()
optimiser=optim.SGD(model.parameters(),lr=0.01)

#scheduler for updating learning rate
#Every 7 epochs learning rate reduced to 10% of curr
step_lr_scheduler=lr_scheduler.StepLR(optimizer=optimiser,step_size=7,gamma=0.1)
model=train_model(model,loss,optimiser,scheduler=step_lr_scheduler,num_epochs=20)