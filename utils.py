
from   collections import OrderedDict
import matplotlib.pyplot as plt
import os
import shutil


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

def plot_sensitivity(list_valid,pathSave):
    fig            = plt.figure()
    ax             = fig.add_subplot(111)
    # txt write
    ax.set_title   ("Sensitivity plot on validset")
    ax.plot        (list_valid, '-',  label="The sensitivity of RNA COVID-2019 identification",color='r')
    ax.set_ylabel  ('Sensitivity ')
    ax.set_xlabel  ("Epochs")
    ax.legend      (loc='lower right')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label        = OrderedDict(zip(labels, handles))
    plt.legend     (by_label.values(), by_label.keys())
    fig.savefig    (os.path.join  (pathSave, "valid.jpg"))


def plot_loss(list_train,pathSave):
    fig            = plt.figure()
    ax             = fig.add_subplot(111)
    # txt write
    ax.set_title   ("MSE loss plot on trainset")
    ax.plot        (list_train, '-',  label="Train loss",color='r')
    ax.set_ylabel  ('MSE')
    ax.set_xlabel  ("Epochs")
    ax.legend      (loc='lower right')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label        = OrderedDict(zip(labels, handles))
    plt.legend     (by_label.values(), by_label.keys())
    fig.savefig    (os.path.join  (pathSave, "train.jpg"))



