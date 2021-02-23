#Command to run tensorboard ----> tensorboard --logdir=name_of_logging_directory(name is "runs" in this case)
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
writer=SummaryWriter("runs/mnist")

#Configuration for the device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size=784
hidden_size=100
num_epochs=3
num_classes=10
batch_size=100
learning_rate=0.01

train_dataset=torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)

test_dataset=torchvision.datasets.MNIST(root='./data',transform=transforms.ToTensor())

print(type(test_dataset))

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)

#Printing sample images
examples=iter(train_loader)
samples,labels=examples.next()
# print(samples.shape,labels.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap='gray')

image_grid=torchvision.utils.make_grid(samples,nrow=10)
writer.add_image('MNIST SAMPLE IMAGES',image_grid)
# writer.close()
# sys.exit()
class Model(nn.Module):
    def __init__(self,in_size,hidden_size,num_classes):
        super(Model,self).__init__()
        self.lin1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.lin2=nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out=self.lin1(x)
        out=self.relu(out)
        out=self.lin2(out)
        return out

model=Model(in_size=input_size,hidden_size=hidden_size,num_classes=num_classes)

loss=nn.CrossEntropyLoss()
optimiser=torch.optim.Adam(model.parameters(),lr=learning_rate)

writer.add_graph(model,samples.view(-1,784))
# writer.close()
# sys.exit()

n_iters=len(train_loader)
running_loss=0.0
running_correct_pred=0
running_total=0
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        #Reshaping the input images
        images=images.view(-1,784).to(device)
        labels=labels.to(device)

        #Forward prop
        outputs=model(images)
        l=loss(outputs,labels)

        #Backward Prop
        optimiser.zero_grad()
        l.backward()
        optimiser.step()

        running_loss+=l.item()
        _,y_pred=torch.max(outputs,1)
        running_correct_pred+=(y_pred==labels).sum().item()
        running_total+=labels.numpy().shape[0]
        #Printing Progress
        if i%100==0:
            print(f'epoch {epoch+1}/{num_epochs},batch-number {i+1}/{n_iters},loss={l.item():.3f}')
            writer.add_scalar('Training Loss',running_loss/100,epoch*n_iters+i)
            writer.add_scalar('Accuracy',running_correct_pred/running_total,epoch*n_iters+i)
            running_loss=0.0
            running_correct_pred=0
            running_total=0


with torch.no_grad():
    n_samples=0
    num_correct_samples=0
    for i,(images,labels) in enumerate(test_loader):
        images=images.view(-1,784).to(device)
        labels=labels.to(device)
        outputs=model(images)
        _,y_pred=torch.max(outputs,1)
        n_samples+=labels.shape[0]
        num_correct_samples+=(y_pred==labels).sum().item()
    accuracy=(num_correct_samples/n_samples)*100
    print(f'Accuracy on test data is {accuracy}')