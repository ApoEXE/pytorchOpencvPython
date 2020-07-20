#NEURAL LAYERS CREATION
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#CREATING NEURAL NETWORK
class Net(nn.Module):
      def __init__(self,data_size):
        self.data_size = data_size
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 60, kernel_size=2, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(60, 1, kernel_size=2, stride=1),
            nn.ReLU())

      def forward(self, x):
         out = self.layer1(x)
         out = self.layer2(out)
         return out 


