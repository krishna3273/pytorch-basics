import torch

x=torch.tensor([1,2,3,4],dtype=torch.float32)
y=torch.tensor([3,6,9,12],dtype=torch.float32)

w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
print(w)

def forward(x):
    return w*x

def loss(y,y_pred):
    return ((y_pred-y)**2).mean()

print(f'Intial prediction:f(10)={forward(10):.3f}')

l_rate=0.01
num_epochs=50

for  epoch in range(num_epochs):
    #Forward prop
    y_pred=forward(x)

    l=loss(y,y_pred)

    l.backward()
    with torch.no_grad():
        w-=l_rate*w.grad
    print(w.grad)
    w.grad.zero_() 
    if(epoch%5):
        print(f'Prediction at epoch {epoch}/{num_epochs}:f(10)={forward(10):.3f}')


