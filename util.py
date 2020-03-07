
from   collections import OrderedDict
import matplotlib.pyplot as plt



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

def plot(list_valid,pathSave):
    fig            = plt.figure()
    ax             = fig.add_subplot(111)
    # txt write
    ax.set_title   ("Error valid plot")
    ax.plot        (list_valid, '-',  label="Error valid",color='r')
    ax.set_ylabel  ('Error to identify two equal RNA sequences')
    ax.set_xlabel  ("Epochs")
    ax.legend      (loc='lower right')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label        = OrderedDict(zip(labels, handles))
    plt.legend     (by_label.values(), by_label.keys())
    fig.savefig    (os.path.join  (pathSave, "valid.jpg"))