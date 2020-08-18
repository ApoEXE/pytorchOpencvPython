import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import os
import glob


dataset = glob.glob("obj/*")

for item in dataset:
    item = item.replace(" ","\ ")
    item2 = item.replace("\ ","")
    item2 = item2.replace("obj/","all/")
    os.system("mv "+str(item)+ " "+item2)
    print(item)
    print(item2)

print("CLEAN")
