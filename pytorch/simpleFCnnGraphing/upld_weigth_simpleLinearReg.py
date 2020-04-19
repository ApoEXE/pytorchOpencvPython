#NEURAL LAYERS CREATION
from net_simplefcNN import *
import torch
import torch.nn as nn
import torch.nn.functional as F
#TRAINING IMPORT
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plotting(x_var,y_var,predit_var):
    plt.plot(x_var,y_var,'bo',x_var,predit_var,'g^')
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
        plotting(array_in[0],array_t[0],array_o[0])



#USING Y=X^2
x_t = [x for x in range(21,32)]
data_size = len(x_t)
x_t = torch.tensor([x_t]).float()
y_t = np.array([[x**2 for x in range(21,32)]])
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
print("data_size",data_size)
net = Net(data_size)
net.load_state_dict(torch.load("Y_Xpwr2_weight.pt"))
net.eval()
print(net)
optimizer = optim.Adam(net.parameters(), lr=1e-2)
test(optimizer,net,x_t,y_t)
