import torch
import torch.nn as nn
from  torch.optim import lr_scheduler


lr=0.1
model=nn.Linear(10,1)

optimiser=torch.optim.Adam(model.parameters(),lr=lr)

# lambda_func=lambda epoch:epoch/10
# scheduler=lr_scheduler.LambdaLR(optimiser,lambda_func)

# lambda_func=lambda epoch:0.95
# scheduler=lr_scheduler.MultiplicativeLR(optimiser,lambda_func)

# scheduler=lr_scheduler.StepLR(optimiser,step_size=2,gamma=0.1)

scheduler=lr_scheduler.MultiStepLR(optimiser,milestones=[2,6],gamma=0.1)

#other examples- Exponential lr,ReduceLROnPlateau
print(optimiser.state_dict())

for epoch in range(10):
    optimiser.step()

    scheduler.step()

    print(optimiser.state_dict()['param_groups'][0]['lr'])