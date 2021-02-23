import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import load_data,letter_to_tensor,line_to_tensor,random_training_example,ALL_LETTERS

NUM_LETTERS=len(ALL_LETTERS)


class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN,self).__init__()
        self.hidden_size=hidden_size
        self.i2h=nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o=nn.Linear(input_size+hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=1)
        self.tanh=nn.Tanh()

    def forward(self,i,h):
        combined=torch.cat((i,h),1)
        hidden=self.tanh(self.i2h(combined))
        combined=torch.cat((i,hidden),1)
        output=self.i2o(combined)
        output=self.softmax(output)
        return output,hidden

    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)


category_data,all_categories=load_data()
num_categories=len(all_categories)

num_hidden=128
rnn=RNN(NUM_LETTERS,num_hidden,num_categories)

#one step
input_tensor=letter_to_tensor('A')
hidden_tensor=rnn.init_hidden()
output,next_hidden=rnn(input_tensor,hidden_tensor)
# print(input_tensor.shape,hidden_tensor.shape,output.shape,next_hidden.shape)

#whole sequence

input_tensor=line_to_tensor('Albert')
hidden_tensor=rnn.init_hidden()
output,next_hidden=rnn(input_tensor[0],hidden_tensor)
# print(input_tensor.shape,hidden_tensor.shape,output.shape,next_hidden.shape)

def category_from_output(output):
    idx=torch.argmax(output).item()
    return all_categories[idx]

# print(category_from_output(output))

loss_criterion=nn.NLLLoss()
learning_rate=0.005
optimiser=torch.optim.SGD(rnn.parameters(),lr=learning_rate)


def train(line_tensor,category_tensor):
    hidden=rnn.init_hidden()

    for i in range(line_tensor.shape[0]):
        output,hidden=rnn(line_tensor[i],hidden)

    loss=loss_criterion(output,category_tensor)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return output,loss.item()


current_loss=0
all_losses=[]
plot_steps,print_steps=1000,5000
num_iters=100_000
for i in range(num_iters):
    category,line,category_tensor,line_tensor=random_training_example(category_data,all_categories)

    output,loss=train(line_tensor,category_tensor)
    current_loss+=loss

    if (i+1)%plot_steps==0:
        all_losses.append(current_loss/plot_steps)
        current_loss=0
        
    if (i+1)%print_steps==0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i+1} {(i+1)/num_iters*100} {loss:.4f} {line} / {guess} {correct}")


plt.figure()
plt.plot(all_losses)
plt.show()

def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        
        hidden = rnn.init_hidden()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        
        guess = category_from_output(output)
        print(guess)


while True:
    sentence = input("Input:")
    if sentence == "quit":
        break
    
    predict(sentence)
