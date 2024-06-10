import os, shutil
from glob import glob
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

def clean_train_test(clss):
    tst = "./output/test/" + clss + "/*"
    train = "./output/train/" + clss + "/*"
    tst_files, train_files = glob(tst), glob(train)
    for f in tst_files:
        os.remove(f)
    for f in train_files:
        os.remove(f)

def create_train_test():
    rootdir = "./output"
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
        os.makedirs(rootdir + "/train")
        os.makedirs(rootdir + "/test")

    classes = ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "ragno", "scoiattolo"]
    for clss in classes:
        clss_source = "./animals10/raw-img/" + clss
        clean_train_test(clss)

        
        allFileNames = os.listdir(clss_source)
        np.random.shuffle(allFileNames)
        test_ratio = 0.3
        train_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)* (1 - test_ratio))])
        train_FileNames = [clss_source+'/'+ name for name in train_FileNames.tolist()]
        test_FileNames = [clss_source+'/' + name for name in test_FileNames.tolist()]

        for name in train_FileNames:
            shutil.copy(name, rootdir +'/train/' + clss)

        for name in test_FileNames:
            shutil.copy(name, rootdir +'/test/' + clss)

translate = {"cane": "dog", "dog": "cane", 
                "cavallo": "horse", "horse" : "cavallo",
                "elefante": "elephant", "elephant" : "elefante", 
                "farfalla": "butterfly", "butterfly": "farfalla", 
                "gallina": "chicken", "chicken": "gallina", 
                "gatto": "cat", "cat": "gatto", 
                "mucca": "cow", "cow": "mucca", 
                "pecora": "sheep", "sheep" : "pecora",
                "ragno" : "spider", "spider": "ragno",
                "scoiattolo": "squirrel", "squirrel": "scoiattolo"}
# create_train_test()

tfms = transforms.Compose([transforms.Resize((512,512)), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.458, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
train = ImageFolder("output/train", tfms)
test = ImageFolder("output/test", tfms)

print(train)
print(test)



train_data_loader = DataLoader(train, batch_size=64, num_workers=4)
valid_data_loader = DataLoader(test, batch_size = 64, num_workers = 4)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(3, 8, 3), # inp (3, 512, 512)
            nn.Conv2d(8, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU() # op (16, 256, 256)
        )
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5), # inp (16, 256, 256)
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(4),
            nn.ReLU() # op (32, 64, 64)
        )
        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3), # inp (32, 64, 64)
            nn.Conv2d(64, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU() # op (64, 32, 32)
        )
        self.ConvLayer4 = nn.Sequential(
            nn.Conv2d(64, 128, 5), # inp (64, 32, 32)
            nn.Conv2d(128, 128, 3),
            nn.MaxPool2d(2),
            nn.ReLU() # op (128, 16, 16)
        )
        self.Lin1 = nn.Linear(15488, 15)
        self.Lin2 = nn.Linear(1500, 150)
        self.Lin3 = nn.Linear(150, 15)
        
        
    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        x = x.view(x.size(0), -1)
        x = self.Lin1(x)
       
        
        return F.log_softmax(x, dim = 1)

