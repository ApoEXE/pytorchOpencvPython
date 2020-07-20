#NEURAL LAYERS CREATION
import torch
import torch.nn as nn
import torch.nn.functional as F
#TRAINING IMPORT
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Net(nn.Module):
        def __init__(self):
            super(Net,self).__init__()
            self.hl = nn.Linear(11,10)#layer one  input 1, output 10
            self.ol = nn.Linear(10,11)# layer 2  input 10 output 1
            #define activation
            self.relu = nn.ReLU() #activation neuron
        def forward(self,x):
            hidden = self.hl(x)
            activation = self.relu(hidden)
            output = self.ol(activation)
            return output
net = Net()

plt.style.use('fivethirtyeight')
#plt.axis([0, 10, 0, 1])

def animate(x_values,y_values):
    x_values.append(next(index))
    y_values.append(random.randint(0, 5))
    plt.cla()
    plt.plot(x_values, y_values)
    
    
def training_loop(n_epochs,optimizer,model,loss_fn,inputs,target):
    for epoch in range(1,n_epochs+1):
        optimizer.zero_grad()#calculate output
        output = net(inputs)#calculate output
        
        loss = loss_fn(output,target)#calculate loss

        loss.backward() # calculate gradient
        optimizer.step()     # update parameters
        if epoch % 2 == 0:
            #print("OUTPUT:")
            array_in = inputs.numpy()
            array_t = target.numpy()
            array_o = output.detach().numpy()
            #print(array_in[0][0])
            #print(array_t[0][0])
            #print(array_o[0][0])
            plt.cla()
            plt.plot(array_in[0],array_t[0],'',array_in[0],array_o[0],'r--')
            plt.pause(0.5)
            print('Epoch %d, Loss %f' % (epoch,float(loss)))
        if float(loss) <0.00001:
        	break
    plt.show()
    return loss
optimizer = optim.Adam(net.parameters(), lr=1e-2)
#x_t = np.array([x for x in range(1,12)])
x_t = [x for x in range(1,12)]
#print(len(x_t))
x_t = torch.tensor([x_t]).float()
#x_t=torch.randn(1, 12)
#x_t = x_t/100
#x_t=torch.randn(1, 12)
#x_t = torch.from_numpy(x_t.values)
#print(x_t)
#print(x_t.size())
y_t = np.array([[x**2 for x in range(1,12)]])
#y_t = torch.randn(1, 1)
#y_t = y_t/100
y_t = torch.from_numpy(y_t).float()
#y_t = torch.randn(1, 12)
#print(y_t)
#print(y_t.size())
#target = torch.from_numpy(_y)
training_loop(n_epochs = 1500, optimizer=optimizer,model=net,loss_fn=nn.MSELoss(),inputs=x_t,target=y_t)
#ani = FuncAnimation(plt.gcf(), animate, 1000)
#plt.tight_layout()
#plt.show()