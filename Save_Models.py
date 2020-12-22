import torch 
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self,num_features):
        super(DummyModel,self).__init__()
        self.lin=nn.Linear(num_features,1)

    def forward(self,x):
        return torch.sigmoid(self.lin(x))


model=DummyModel(6)
FILE_NAME="dummy_model.pth"




#Saving a model(method-1)
# torch.save(model,FILE_NAME)

#Loading a Model(method-1)
# model=torch.load(FILE_NAME)
# model.eval()


#Saving models(method-2)
torch.save(model.state_dict(),FILE_NAME)

#Loading models(method-2)
loaded_model=DummyModel(6)
loaded_model.load_state_dict(torch.load(FILE_NAME))
model.eval()

for param in model.parameters():
    print(param)


#Saving a checkpoint during training
learning_rate=0.001
optimiser=torch.optim.SGD(model.parameters(),lr=learning_rate)

checkpoint={
    "curr_epoch":50,
    "model_state":model.state_dict(),
    "optim_state":optimiser.state_dict()
}

torch.save(checkpoint,"checkpoint.pth")

#Loading checkpoint again
model = DummyModel(6)
optimiser=torch.optim.SGD(model.parameters(), lr=0)

checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint['model_state'])
optimiser.load_state_dict(checkpoint['optim_state'])
epoch = checkpoint['epoch']

#use map_loacation in torch.load to map cpu saved model on gpu or vice-versa