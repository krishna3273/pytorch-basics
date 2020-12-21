import numpy as np

x=np.array([1,2,3],dtype=np.float64)
y=np.array([3,6,9],dtype=np.float64)

w=0.0

def forward(x):
    return w*x

def loss(y,y_pred):
    return ((y_pred-y)**2).mean()

def grad(x,y,y_pred):
    return np.dot(2*x,y_pred-y).mean()

print(f'Intial prediction:f(10)={forward(10):.3f}')

l_rate=0.01
num_epochs=20

for  epoch in range(num_epochs):
    #Forward prop
    y_pred=forward(x)

    l=loss(y,y_pred)

    dw=grad(x,y,y_pred)

    w=w-dw*l_rate

    print(f'Prediction at epoch {epoch}/{num_epochs}:f(10)={forward(10):.3f}')


