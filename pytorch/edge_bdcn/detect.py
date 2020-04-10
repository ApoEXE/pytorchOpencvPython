import os
import cv2
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import time
from PIL import Image
#NEURAL LAYERS CREATION
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#TRAINING IMPORT
import torch.optim as optim



#class
import bdcn

def test(model):
    
    mean_bgr = np.array([104.00699, 116.66877, 122.67892])
    #mean_bgr = np.array([100, 100, 100])
    model.eval()
    start_time = time.time()
    all_t = 0
    imagen = cv2.imread("img1.jpg")
    heigh=imagen.shape[0]
    weigh=imagen.shape[1]
    print("H:",heigh,"W:",weigh)
        # print(os.path.join(test_root, nm))
        # data = cv2.resize(data, (data.shape[1]/2, data.shape[0]/2), interpolation=cv2.INTER_LINEAR)
    data = np.array(imagen, np.float32)
    data -= mean_bgr
    data = data.transpose((2, 0, 1))
    data = torch.from_numpy(data).float().unsqueeze(0)
    data = Variable(data)

    out = model(data)
    
    out = [torch.sigmoid(x).cpu().data.numpy()[0, 0] for x in out]
    img = out[-1]
    img = 255*img
    #print(img)
    print('Overall Time use: ', time.time() - start_time)
    #a = np.reshape(out,weigh,heigh)
    #print(type(img))

    cv2.imshow("out",img)
    cv2.waitKey()

if __name__ == '__main__':
	model = bdcn.BDCN()
	model.load_state_dict(torch.load("bdcn_pretrained_on_bsds500.pth", map_location=torch.device('cpu')))
	model.eval()
	#print(model)
	test(model)
