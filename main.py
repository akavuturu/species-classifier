import numpy as np
import tensorflow as tf
import os
import splitfolders
# 60 train, 20 val, 20 test

def parse_dataset():
    data_path = "animals10"
    splitfolders.ratio(data_path, output_path="train/test", ratio=(0.7,0.3))



if __name__ == "main":
    parse_dataset()
    