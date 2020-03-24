import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def createList(r1, r2): 
    return [item for item in range(r1, r2+1)] 

# Define some points:
fileName  = "/home/nick/Desktop/results/alexnet/valid.txt"
clean     = []
with open(fileName, 'r') as f:
    lines = f.readlines()
for l in lines:
    clean.append(float(l.replace("\n", "")))  
points   = np.array(clean).T
y = points
x = createList(0,len(points)-1)

'''
import numpy as np
# Interpolate it to new time points
from scipy.interpolate import interp1d
#########################################
linear_interp = interp1d(x, y, kind='cubic')
xnew = np.arange(0,100)
linear_results = linear_interp(xnew)

'''



y_av = movingaverage(y, 20)

# Plot the data and the interpolation
from matplotlib import pyplot as plt
plt.plot(x, y_av, label='linear interp')
plt.legend()
plt.show()