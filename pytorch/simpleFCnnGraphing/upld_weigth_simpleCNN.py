#NEURAL LAYERS CREATION
from net_simpleCNN import *
import torch
import torch.nn as nn
import torch.nn.functional as F
#TRAINING IMPORT
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plotting(x_var,y_var,predit_var):
    plt.plot(x_var,y_var,'bo',x_var,predit_var,'r^')
    plt.show()

def test(optimizer,model,inputs,target):
    #VALIDATING DATA
    print("TESTING ACCURACY OF net")
    correct = 0
    total = 0
    with torch.no_grad():
        output = net(inputs) # returns a list
        array_in = inputs.numpy()
        array_t = target.numpy()
        array_o = output.detach().numpy()
        error=abs(array_t[0]-array_o[0])
        error = np.mean(error)
        print("model Error",error)
        print(array_t[0])
        print(array_o[0].astype(int))
        plotting(array_in[0],array_t[0],array_o[0])



muestra = 5
#USING Y=X^2
x_t = np.random.randint(low=1, high=60, size=muestra)
data_size = len(x_t)
y_t = np.array([[x**2 for x in x_t]])
x_t = torch.tensor([x_t]).float()
y_t = torch.from_numpy(y_t).float()
x_t = x_t.view([1, 1, data_size])
y_t = y_t.view([1, 1, data_size])
#print(x_t)
#print(y_t)

print("data_size",data_size)
net = Net(data_size)
net.load_state_dict(torch.load("Y_Xpwr2_weight.pt"))
net.eval()
print(net)
optimizer = optim.Adam(net.parameters(), lr=1e-2)
test(optimizer,net,x_t,y_t)
