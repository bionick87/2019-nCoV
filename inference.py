import torch
import torchvision
import torchvision.datasets as dset
from   torchvision      import transforms
from   dataset          import Dataset
from   torch.utils.data import DataLoader
from   torch.autograd   import Variable
import matplotlib.pyplot as plt
from   model            import SiameseNet
import time
import numpy            as np
import gflags 
import sys
import os
from   tqdm import tqdm
from   utils import *
import random
import math


def inference(path):
    model = SiameseNet()
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    print(model)


if __name__== "__main__":
    path_model = "/Users/nicolosavioli/Desktop/alexnet_2019-nCoV.pt"
    inference(path_model)
