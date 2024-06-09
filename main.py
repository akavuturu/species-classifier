import numpy as np
# import tensorflow as tf
import os, glob, shutil
import random, math
# import splitfolders

def clean_train_test(clss):
    tst = "./output/test/" + clss + "/*"
    train = "./output/train/" + clss + "/*"
    tst_files, train_files = glob.glob(tst), glob.glob(train)
    for f in tst_files:
        os.remove(f)
    for f in train_files:
        os.remove(f)

def create_train_test():
    source = "./animals10"
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

if __name__ == "__main__":
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