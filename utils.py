
from   collections import OrderedDict
import matplotlib.pyplot as plt
import os
import shutil
import torch
import torchvision
import torchvision.datasets as dset
from   torchvision      import transforms
from   dataset          import Dataset
from   torch.utils.data import DataLoader
from   torch.autograd   import Variable
from   model            import SiameseNet


def write_txt(lossList,lossSavePath):
    if os.path.exists(lossSavePath):
        os.remove(lossSavePath) 
    with open(lossSavePath, "a") as f:
        for d in range(len(lossList)):
            f.write(str(lossList[d]) +"\n")

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

def makeFolder(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    else:
        shutil.rmtree(dirName)
        os.mkdir(dirName)


def loadModel(path):
    model = SiameseNet()
    model.load_state_dict(torch.load(path,map_location=torch.device('cuda')))
    model.cuda()
    model.eval()
    return model

def plot_sensitivity(list_valid,pathSave):
    fig            = plt.figure()
    ax             = fig.add_subplot(111)
    # txt write
    ax.set_title   ("Validset sensitivity")
    ax.plot        (list_valid, '-',  label="Sensitivity",color='r')
    ax.set_ylabel  ('Sensitivity (%)')
    ax.set_xlabel  ("Epochs")
    ax.legend      (loc='lower right')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label        = OrderedDict(zip(labels, handles))
    plt.legend     (by_label.values(), by_label.keys())
    fig.savefig    (os.path.join             (pathSave, "valid.jpg"))
    write_txt      (list_valid,os.path.join  (pathSave, "valid.txt"))


def plot_loss(list_train,pathSave):
    fig            = plt.figure()
    ax             = fig.add_subplot(111)
    # txt write
    ax.set_title   ("Trainset MSE loss")
    ax.plot        (list_train, '-',  label="Loss",color='r')
    ax.set_ylabel  ('MSE')
    ax.set_xlabel  ("Epochs")
    ax.legend      (loc='lower right')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label        = OrderedDict(zip(labels, handles))
    plt.legend     (by_label.values(), by_label.keys())
    fig.savefig    (os.path.join  (pathSave, "train.jpg"))



