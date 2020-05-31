import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
#NEURAL LAYERS CREATION
import torch
import torch.nn as nn
import torch.nn.functional as F
#TRAINING IMPORT
import torch.optim as optim
from torch.autograd import Variable

import sys

if torch.cuda.is_available():  
  dev = "cuda:0" 
  print("running on GPU")
else:  
  dev = "cpu"  
  print("running on CPU")
device = torch.device(dev)
print("how many CUDA devices:",torch.cuda.device_count())

#CREATING NEURAL NETWORK
class Net(nn.Module):
        
    def convs(self, x):
        #F.relu is activation function
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        #flatthen to get into a linear fully connected  neural
        x = torch.flatten(x, 1, -1)
        if self._to_linear is None:#this was created to obtain the output of last conv
            self._to_linear =x.shape[1]
            print("output of conv:",self._to_linear)
        return x
    
    def forward(self, x):
        x = self.convs(x)#convolutions maxpull pass
        x = F.relu(self.fc1(x))#pass the convolution through fully connected linear
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)

    def __init__(self):
        super().__init__()
        #Number of Channel 1,32,64, OUPUT FEATURE 32, 64, 128, KERNEL SIZE 5x5 
        #LAYER 1 CONVOLUTIONAL THIS ARE 2D CONVOLUTIONAL LAYERS
        self.conv1 = nn.Conv2d(1,32,5)
        #LAYER 2 CONVOLUTIONAL THIS ARE 2D CONVOLUTIONAL LAYERS
        self.conv2 = nn.Conv2d(32,64,5)
        #LAYER 3 CONVOLUTIONAL THIS ARE 2D CONVOLUTIONAL LAYERS
        self.conv3 = nn.Conv2d(64,128,5)
        
        #dummy image to get output of conv so fc neurans has a starting input
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        #output of this is the self.to_linear output of conv
        self.convs(x)
        
        #INPUT HxW, how many Neurals to create (its call Dense Layer)
        #LAYER LINEAR 1
        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        #LAYER LINEAR 2
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).

def test(test_x):
    with torch.no_grad():
        outputs = net(test_x)
        #print(outputs)
        
        values = outputs.cpu().detach().numpy()
        print(values)
        
        if values[0][0]>values[0][1]:
            print("CAT")
            prediction = "CAT"
        else:
            print("DOG")
            prediction = "DOG"
    return prediction
        
        


def opencvToPytorch(img):
    #print("opencvToPytorch")
    #print("1#",img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    data = np.array(gray, np.float32)
    #in gray scale there is no channel axis numpy creates that axis in position 2
    data = np.expand_dims(data,2)
    #from [H][W][channel] to [channel][W][H]
    data = data.transpose((2, 0, 1))
    #print(type(data))
    data = data/255.0
    data = torch.from_numpy(data).float().unsqueeze_(0)
    data=Variable(data)
    return data

#--------------------------CODE FROM HERE ON ------------------------------#        
#load image
img_path = str(sys.argv[1])
image = cv2.imread(img_path)
img = cv2.resize(image, (50, 50))
#STEP2: CREATE OUR NEURAL NETWORK  IN A MODULE "CLASS" 
net = Net()  
net.load_state_dict(torch.load("fullcnn.pt"))
net.eval()
net.to(device)
print(net)   
#STEP3: CONVERT IMAGE TO TENSO
test_x = opencvToPytorch(img)
#STEP6: TRAIN WITH TRAINNING DATASET
prediction = test(test_x.to(device))
fontScale = 0.8# Would work best for almost square images
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image,prediction, (int(image.shape[1]/2),image.shape[0]-20),font , fontScale, (0,0,255),thickness)
cv2.imshow("animal",image)
cv2.waitKey(5000)


