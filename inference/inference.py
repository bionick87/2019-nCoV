import torch
import torchvision
import torchvision.datasets as dset
from   torchvision      import transforms
from   torch.utils.data import DataLoader
from   torch.autograd   import Variable
import matplotlib.pyplot as plt
from   model            import SiameseNet
import time
import numpy            as np
import sys
import os
from   tqdm import tqdm
import random
import math
from   PIL import Image
from   torchvision import transforms


def removeMACOS(list):
    if ".DS_Store" in list: list.remove(".DS_Store")
    return list

def getModel(path):
    model = SiameseNet()
    model.load_state_dict(torch.load(path,map_location=torch.device('cuda')))
    model.cuda()
    model.eval()
    return model

def get_inference(model_path,pep_path,hr1_path):
    transform_valid = transforms.Compose([transforms.ToTensor()])
    net             = getModel(model_path)
    selected_pep    = []
    mean_save       = []
    for pep_index in tqdm(removeMACOS(os.listdir(pep_path))):
        pep_name    = os.path.splitext(pep_index)[0]
        pep         = transform_valid(Image.open(os.path.join(pep_path,pep_index)))
        pep_cuda    = Variable(pep.unsqueeze(0)).cuda().float()
        prob_hr1    = [] 
        for hr1_index in removeMACOS(os.listdir(hr1_path)):
            hr1         = transform_valid(Image.open(os.path.join(hr1_path,hr1_index)))
            hr1_cuda    = Variable(hr1.unsqueeze(0)).cuda().float()
            output      = net.forward(pep_cuda, hr1_cuda).data.cpu().numpy()
            prob_hr1.append(output)
        mean_hr1_iteraction = np.mean(prob_hr1)
        mean_save.append(mean_hr1_iteraction)
        selected_pep.append(pep_name)
    ind = mean_save.index(max(mean_save))
    pep = selected_pep[ind]
    print("...High interaction with: " + pep)


if __name__== "__main__":
    path_model = "/home/nick/Desktop/results/alexnet/model_20.pt"
    path_pep   = "/home/nick/Desktop/pepdata"
    path_hr1   = "/home/nick/Desktop/code/2019-nCoV/virus_genome/hr1"
    get_inference(path_model,path_pep,path_hr1)
