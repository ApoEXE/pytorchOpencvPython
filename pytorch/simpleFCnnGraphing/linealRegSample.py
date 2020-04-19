#NEURAL LAYERS CREATION
import torch
import torch.nn as nn
import torch.nn.functional as F
#TRAINING IMPORT
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
        def __init__(self,size_in):
            super(Net,self).__init__()
            self.size_in = size_in
            self.hl = nn.Linear(self.size_in,10)#layer one  input 1, output 10
            self.ol = nn.Linear(10,self.size_in)# layer 2  input 10 output 1
            #define activation
            self.relu = nn.ReLU() #activation neuron
        def forward(self,x):
            hidden = self.hl(x)
            activation = self.relu(hidden)
            output = self.ol(activation)
            return output
    
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

            plotting(array_in[0],array_t[0],array_o[0])
            print('Epoch %d, Loss %f' % (epoch,float(loss)))
        if float(loss) <= 0.003:
            print("FINISHED TRAINNING")
            break
    plt.show()
    torch.save(net.state_dict(), "Y_Xpwr2_weight.pt")
    return loss

def plotting(x_var,y_var,predit_var):
    plt.cla()
    plt.plot(x_var,y_var,'bo',x_var,predit_var,'g^')
    plt.pause(0.1)


x_t = [x for x in range(1,12)]
data_size = len(x_t)
x_t = torch.tensor([x_t]).float()
y_t = np.array([[x**2 for x in range(1,12)]])
y_t = torch.from_numpy(y_t).float()
'''
example = np.array([[2,15],[3,28],[5,42],[13,64],[8,50],[16,90],[11,58],[1,8],[9,54]])
xval = [i[0] for i in example]
print("xval",xval,"size",len(xval))
yval = [i[1] for i in example]
print("yval",yval,"size",len(yval))
x_t = torch.tensor([xval]).float()
y_t = torch.tensor([yval]).float()
print("x_t",x_t,"size",x_t.size())
print("y_t",y_t,"size",y_t.size())
'''

net = Net(data_size)
optimizer = optim.Adam(net.parameters(), lr=1e-2)


training_loop(n_epochs = 1500, optimizer=optimizer,model=net,loss_fn=nn.MSELoss(),inputs=x_t,target=y_t)
