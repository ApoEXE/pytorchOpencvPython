from net_simplefcNN import *
#NEURAL LAYERS CREATION
import torch
import torch.nn as nn
import torch.nn.functional as F
#TRAINING IMPORT
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import time
plt.ylim(0, 100)

    
def training_loop(optimizer,model,loss_fn):
    
    _loss = 10000.
    new_data = False
    epoch = 0
    _last_loss = 10000.
    while _loss > 0.1:
        optimizer.zero_grad()#calculate output
        if new_data == False:
            #MODIFY INPUT
            x_t=np.random.randint(low=1, high=12, size=11)
            #print("x_t:",x_t)
            x_t = torch.tensor([x_t]).float()
            #MODIFY OUTPUT
            y_t = np.array([[x**2 for x in x_t.numpy()]])
            #print("y_t:",y_t)
            y_t = torch.from_numpy(y_t).float()
            #new_data = True

        output = net(x_t)#calculate output
            
        loss = loss_fn(output,y_t)#calculate loss

        loss.backward() # calculate gradient
        optimizer.step()     # update parameters
            
        #print("OUTPUT:")
        array_in = x_t.numpy()
        array_t = y_t.numpy()
        array_o = output.detach().numpy()
        #plotting(array_in[0],array_t[0],array_o[0])
        _loss = float(loss)
        sec = int(time.time())

        print('Loss',_loss)

        #new_data = False
        if(epoch % 12 == 0):
            if _last_loss > _loss:
                plt.ylim(0, _loss*3)
                _last_loss = _loss
                #plt.plot(sec,_loss,'ro')
                #plt.pause(0.00001)
        if(epoch % 1500 == 0):
            print("SAVED",sec)
            torch.save(net.state_dict(), "Y_Xpwr2_weight.pt")
            #if(epoch % 100 == 0):
                #new_data = False
            #print('Epoch %d, Loss %f' % (epoch,float(loss)))
        epoch += 1
    #plt.show()
    print("FINISHED")
    return _loss

def plotting(x_var,y_var,predit_var):
    plt.cla()
    plt.plot(x_var,y_var,'bo',x_var,predit_var,'g^')
    plt.pause(0.1)



data_size = 11
net = Net(data_size)
optimizer = optim.Adam(net.parameters(), lr=1e-2)


training_loop(optimizer=optimizer,model=net,loss_fn=nn.MSELoss())
