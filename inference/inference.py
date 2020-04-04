import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as dset
from   torchvision      import transforms
from   torch.utils.data import DataLoader
from   torch.autograd   import Variable
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
from   dataset          import Dataset
from   torch.utils.data import DataLoader

def removeMACOS(list):
    if ".DS_Store" in list: list.remove(".DS_Store")
    return list

def getModel(path):
    model = SiameseNet()
    model.load_state_dict(torch.load(path,map_location=torch.device('cuda')))
    model.cuda()
    model.eval()
    return model

def measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(TP, FP, TN, FN)

def get_test(model_path,train_path,test_path,valid_path,\
             max_iter_train,flag,batch_size=10,workers=4):
    net               = getModel(model_path)
    net.eval()
    print("\n ...Test")
    testSet     = Dataset   (train_path,test_path,valid_path,max_iter_train,"test")
    trainLoader = DataLoader(testSet, batch_size=batch_size, shuffle=False, num_workers=workers)
    cross_valid = [] 
    for cross_validation in tqdm(range(100)):
        sensitivity_valid   = []
        for _, (valid1, valid2, label_valid) in enumerate(trainLoader, 1):
            test1, test2  = valid1.cuda(), valid2.cuda()
            output_net        = net.forward(test1, test2, flag)
            y_actual = []
            y_hat    = []
            for i in range(output_net.size()[0]):
                output_net_np = math.ceil(output_net[i].data.cpu().numpy())
                y_actual.append(1)
                if output_net_np == 1.0 or output_net_np == 1:
                   y_hat.append(1)
                else:
                   y_hat.append(0)
            TP, FP, TN, FN = measure(y_actual, y_hat)   
            sensitivity = 100*(TP/(TP+FN))
            sensitivity_valid.append(sensitivity)
        cross_valid.append(np.mean(sensitivity_valid))
    print("\n ...Sensitivity mean: " + str(np.mean(cross_valid)) + " std: " + str(np.std(cross_valid)))


def get_inference(model_path,pep_path,hr1_path,flag):
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
            output      = net.forward(pep_cuda, hr1_cuda,flag).data.cpu().numpy()
            prob_hr1.append(output)
        mean_hr1_iteraction = np.mean(prob_hr1)
        mean_save.append(mean_hr1_iteraction)
        selected_pep.append(pep_name)
    ind = mean_save.index(max(mean_save))
    pep = selected_pep[ind]
    print("\n ...High interaction peptide: " + pep + " interaction: " + str(max(mean_save)) + "%")

if __name__== "__main__":
    model_path     = "path-to-model/AlexNet_pretrain.pt"
    train_path     = "path-to-datset/train"
    test_path      = "path-to-datset/test"
    valid_path     = "path-to-datset/valid"
    path_pep       = "path-to-pep/pepdata"
    path_hr1       = "./virus_genome/hr1"
    test_phase     = True
    #########################################################################
    if test_phase == True:
        get_inference(model_path,path_pep,path_hr1,True)
    else:
        get_test(model_path,train_path,test_path,valid_path,200,False)
    #########################################################################    
    



