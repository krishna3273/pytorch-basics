import torch

x=torch.randn(3,requires_grad=True)
print(x)

y=x+2
y.retain_grad()
print(y)
z=y*y
print(z)
# out=z.mean()
# print(out)
#v is the vector that should contain derivative of scalar L(typically  loss) w.r.t z
v=torch.tensor([1.1,3.0,4.5])
z.backward(v)
#Prints the derivative of L w.r.t to x
# print(x.grad,y.grad)


#Not track gradients
#Option-1
x.requires_grad_(False)
# print(x)
#Option-2
y=x.detach()
# print(y)
#option-3
with torch.no_grad():
    y=x+2
    # print(y)

#Demo model

weights=torch.ones(4,requires_grad=True)

for epochs in range(3):
    out=(weights+2).sum()
    out.backward()
    print(weights.grad)
    weights.grad.zero_()
