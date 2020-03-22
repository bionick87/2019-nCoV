import torch
from   torch.utils.data import Dataset, DataLoader
import os
from   numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from   PIL import Image
from   torchvision import transforms
import cv2
import math


class Dataset(Dataset):
    def __init__(self, dataPathTrain,\
                 dataPathValid,\
                 dataPathTest,iteration,type):

        super(Dataset, self).__init__()
        np.random.seed(0)
        #####################################
        self.transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor()])
        #####################################
        self.transform_v = transforms.Compose([
        transforms.ToTensor()])
        #######################
        self.negTrain,\
        self.posTrain = self.loadData(dataPathTrain)
        self.negTrain = self.removeMACOS(self.negTrain)
        self.posTrain = self.removeMACOS(self.posTrain)

        self.negValid,\
        self.posValid = self.loadData(dataPathValid)
        self.negValid = self.removeMACOS(self.negValid)
        self.posValid = self.removeMACOS(self.posValid)

        self.negTest,\
        self.posTest  = self.loadData(dataPathTest)
        self.negTest  = self.removeMACOS(self.negTest)
        self.posTest  = self.removeMACOS(self.posTest)
        #######################
        self.dataPathTrain = dataPathTrain
        self.dataPathValid = dataPathValid
        self.dataPathTest  = dataPathTest
        self.type          = type
        self.iter          = iteration
    
    def removeMACOS(self,list):
        if ".DS_Store" in list: list.remove(".DS_Store")
        return list

    def loadData(self, dataPath):
        pos = os.listdir(os.path.join(dataPath,"negative"))
        neg = os.listdir(os.path.join(dataPath,"positive"))
        return pos,neg

    
    def __len__(self):
        return self.iter

    def __getitem__(self, index):
        neg_data  = None
        pos_data  = None
        image1    = None
        image2    = None
        label     = 0 
        if   self.type == "train":
              index_neg = random.randint(0, len(self.negTrain)-1) 
              index_pos = random.randint(0, len(self.posTrain)-1)
              neg_data  = self.transform(Image.open(os.path.join(self.dataPathTrain,"negative",self.negTrain[index_neg])))
              pos_data  = self.transform(Image.open(os.path.join(self.dataPathTrain,"positive",self.posTrain[index_pos])))
        elif self.type == "valid":
              index_neg = random.randint(0, len(self.negValid)-1) 
              index_pos = random.randint(0, len(self.posValid)-1)
              neg_data  = self.transform_v(Image.open(os.path.join(self.dataPathTrain,"negative",self.negValid[index_neg])))
              pos_data  = self.transform_v(Image.open(os.path.join(self.dataPathTrain,"positive",self.posValid[index_pos])))
        elif self.type == "test":
              index_neg = random.randint(0, len(self.negTest)-1) 
              index_pos = random.randint(0, len(self.posTest)-1)
              neg_data  = self.transform_v(Image.open(os.path.join(self.dataPathTrain,"negative",self.negTest[index_neg])))
              pos_data  = self.transform_v(Image.open(os.path.join(self.dataPathTrain,"positive",self.posTest[index_pos])))
        
        if  self.type == "train":
              random_select = random.randint(0, 1)
              if random_select == 0:
                 image1 = pos_data
                 image2 = pos_data
                 label  = torch.from_numpy(np.array([0.0], dtype=np.float32))
              else:
                 image1 = pos_data
                 image2 = neg_data
                 label  = torch.from_numpy(np.array([1.0], dtype=np.float32))
        else:
                 image1 = pos_data
                 image2 = neg_data
        return image1, image2, label



    


