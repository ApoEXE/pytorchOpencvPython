#NEURAL LAYERS CREATION
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):

        def __init__(self,size_in):
            super(Net,self).__init__()
            self.size_in = size_in
            self.mult = 6
            self.nnSize = size_in*self.mult
            self.hl1 = nn.Linear(self.size_in,self.nnSize)#layer one  input 1, output 10
            self.hl2 = nn.Linear(self.nnSize,self.nnSize)#layer one  input 1, output 10
            self.hl3 = nn.Linear(self.nnSize,self.nnSize)#layer one  input 1, output 10
            self.ol = nn.Linear(self.nnSize,self.size_in)# layer 2  input 10 output 1
            #define activation
            self.relu = nn.ReLU() #activation neuron
        def forward(self,x):
            hidden = self.hl1(x)
            activation = self.relu(hidden)


            hidden = self.hl2(activation)
            activation = self.relu(hidden)
            hidden = self.hl3(activation)
            activation = self.relu(hidden)

            output = self.ol(activation)
            return output