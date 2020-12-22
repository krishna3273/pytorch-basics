import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Helper function to plot normalised images as normal  ones
def imshow(img):
    img = img / 2 + 0.5  #Removing the normalisation applied before plotting
    npimg = img.numpy() #torch tensor to numpy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


num_epochs=2
batch_size=10
learning_rate=0.01

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#Downloading datasets and reading
train_dataset=torchvision.datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)
test_dataset=torchvision.datasets.CIFAR10(root='./data',transform=transform,train=False,download=True)

#Loading the datasetby dividing into batches
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)

classes = np.array(['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #conv2d(num_input_channels,num_output_channels,num_filters)
        self.conv1=nn.Conv2d(3,6,5)
        #MaxPool2d(filter_size,stride)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        #16*5*5 units =400 units is the shapeof resulting image after conv layers and pooling layers
        self.fc1=nn.Linear(400,120)
        self.fc2=nn.Linear(120,80)
        self.fc3=nn.Linear(80,10)
    def forward(self,x):
        out=F.relu(self.conv1(x))
        out=self.pool(out)
        out=F.relu(self.conv2(out))
        out=self.pool(out)
        out=out.view(-1,400)
        out=F.relu(self.fc1(out))
        out=F.relu(self.fc2(out))
        #No activation in final layer because softmax included in cross-entropy loss already
        out=self.fc3(out)
        return out

model=CNN().to(device)
loss=nn.CrossEntropyLoss()
optimiser=torch.optim.SGD(model.parameters(),lr=learning_rate)

n_iters=len(train_loader)

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        #Reshaping the input images
        images=images.to(device)
        labels=labels.to(device)

        #Forward prop
        outputs=model(images)
        l=loss(outputs,labels)

        #Backward Prop
        optimiser.zero_grad()
        l.backward()
        optimiser.step()

        #Printing Progress
        if i%500==0:
            print(f'epoch {epoch+1}/{num_epochs},batch-number {i+1}/{n_iters},loss={l.item():.3f}')


with torch.no_grad():
    num_correct = 0
    num_total = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        num_total += labels.size(0)
        num_correct += (predicted == labels).sum().item()
        labels_size=labels.numpy().shape[0]
        for i in range(labels_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * num_correct / num_total
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')