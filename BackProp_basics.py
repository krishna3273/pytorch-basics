import torch

# x=torch.randn(1,requires_grad=True)
# y=torch.randn(1,requires_grad=True)

# print(x)
# print(y)

# z=x*y

# z.backward()

# print(x.grad)
# print(y.grad)

x=torch.tensor(1.0)
w=torch.tensor(1.0,requires_grad=True)
y_req=torch.tensor(2.0)

#Forward pass
y_curr=w*x
loss=(y_curr-y_req)**2

print(loss)

#Backward prop
loss.backward()
print(w.grad)