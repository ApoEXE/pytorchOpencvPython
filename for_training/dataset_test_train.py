import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import os
import glob


dataset = glob.glob("obj/*.jpg")



x_train,x_test=train_test_split(dataset,test_size=0.2)

with open('valid.txt', 'w') as f:
    for item in x_test:
    	#print(item)
    	item = item.replace(" .",".")
    	#print(item)
        os.system("cp "+item+" "+"valid/")
        txt = item.replace(".jpg",".txt")
        print(txt)
        os.system("cp "+txt+" valid/")
        item = item.replace("obj/","data/obj/")
        f.write("%s\n" % item)
        #print(item)
        
with open('train.txt', 'w') as f:
    for item in x_train:
    	item = item.replace(" .",".")
        os.system("cp "+item+" "+"train/")
        txt = item.replace(".jpg",".txt")
        os.system("cp "+txt+" train/")
        item = item.replace("obj/","data/obj/")
        f.write("%s\n" % item)
        #print(item)


print("SAVED")
print("test: "+str(len(x_test)))
print("train: "+str(len(x_train)))
