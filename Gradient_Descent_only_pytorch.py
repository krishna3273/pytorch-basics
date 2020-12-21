import torch
import torch.nn as nn
x=torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y=torch.tensor([[3],[6],[9],[12]],dtype=torch.float32)

num_samples,num_features=x.shape

model=nn.Linear(num_features,num_features)

x_test=torch.tensor([10.0],dtype=torch.float32)

print(f'Intial prediction:f(10)={model(x_test).item():.3f}')

l_rate=0.05
num_epochs=50
loss=nn.MSELoss()
optimiser=torch.optim.SGD(model.parameters(),lr=l_rate)
for  epoch in range(num_epochs):
    #Forward prop
    y_pred=model(x)

    l=loss(y,y_pred)

    l.backward()

    optimiser.step()
    optimiser.zero_grad() 
    if(epoch%5):
        print(f'Prediction at epoch {epoch}/{num_epochs}:f(10)={model(x_test).item():.3f}')


