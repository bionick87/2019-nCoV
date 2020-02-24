
from   collections import OrderedDict
import matplotlib.pyplot as plt



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