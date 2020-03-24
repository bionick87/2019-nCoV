import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import collections


def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def createList(r1, r2): 
    return [item for item in range(r1, r2+1)] 

def getPoints(fileName):
    clean     = []
    with open(fileName, 'r') as f:
        lines = f.readlines()
    for l in lines:
        clean.append(float(l.replace("\n", "")))  
    points   = np.array(clean).T
    x    = createList(0,len(points)-1)
    y_av = movingaverage(points, 10)
    return  y_av


#####################################################################
file_alexnet        = "/home/nick/Desktop/results/alexnet/valid.txt"
file_alexnet_np     = "/home/nick/Desktop/results/alexnet_no_pretrain/valid.txt"
file_save           = "/home/nick/Desktop/code/2019-nCoV/paper/plot/plot.png"
#####################################################################
alexnet_pretrain    = getPoints(file_alexnet)
alexnet_no_pretrain = getPoints(file_alexnet_np) 


print(alexnet_pretrain)
#####################################################################
fig            = plt.figure()
ax             = fig.add_subplot(111)
#ax.set_title   ("Validset sensitivity")
ax.plot        (alexnet_pretrain, '-',  label="Alexnet pretrain",color='r')
ax.plot([np.mean(alexnet_pretrain)]*len(alexnet_pretrain), linestyle='--', color='r')


ax.plot        (alexnet_no_pretrain, '-',  label="Alexnet",color='g')
ax.plot        ([np.mean(alexnet_no_pretrain)]*len(alexnet_no_pretrain), linestyle='--', color='g')


ax.set_ylabel  ('Sensitivity (%)')
ax.set_xlabel  ("Epochs (x10)")
ax.legend      (loc='lower right')
handles, labels = plt.gca().get_legend_handles_labels()
by_label        = collections.OrderedDict(zip(labels, handles))
plt.legend     (by_label.values(), by_label.keys())
fig.savefig    (file_save)

