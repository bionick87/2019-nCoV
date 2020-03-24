import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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
    y_av = movingaverage(points, 20)
    return x,y_av






fig            = plt.figure()
ax             = fig.add_subplot(111)
#ax.set_title   ("Validset sensitivity")
ax.plot        (alexnet_pretrain, '-',  label="Sensitivity",color='r')
ax.plot        (alexnet_no_pretrain, '-',  label="Sensitivity",color='r')
ax.set_ylabel  ('Sensitivity (%)')
ax.set_xlabel  ("Epochs (x100)")
ax.legend      (loc='lower right')
handles, labels = plt.gca().get_legend_handles_labels()
by_label        = OrderedDict(zip(labels, handles))
plt.legend     (by_label.values(), by_label.keys())
fig.savefig    (os.path.join             (pathSave, "valid.jpg"))

