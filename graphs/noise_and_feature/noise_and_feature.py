import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import glob, os
import csv
import numpy as np

x=[]
y=[]
ymin=[]
ymax=[]

k=[]
kmin=[]
kmax=[]

l=[]
lmin=[]
lmax=[]

os.chdir("/home/smu/Desktop/RNN/graphs/noise_and_feature")

with open('data_both_nonoise.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        x.append(str(row[0]))
        ymin.append(float(row[8])-float(row[6]))
        ymax.append(float(row[7])-float(row[8]))
        y.append(float(row[8]))

yerror = [ymin, ymax]

with open('data_both_envnoise.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        k.append(float(row[8]))
        kmin.append(float(row[8])-float(row[6]))
        kmax.append(float(row[7])-float(row[8]))

kerror = [kmin, kmax]

with open('data_both_mixednoise.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        l.append(float(row[8]))
        lmin.append(float(row[8])-float(row[6]))
        lmax.append(float(row[7])-float(row[8]))

lerror = [lmin, lmax]

y_pos = np.arange(len(x))

plt.xlim(18,90)
# Create bars
fig1 = plt.barh(y_pos, y, xerr=yerror, height=0.25)
fig2 = plt.barh(y_pos+0.25, k, xerr=kerror, height=0.25)
fig3 = plt.barh(y_pos+0.5, l, xerr=lerror, height=0.25)

# Create names on the x-axis
plt.yticks(y_pos+0.25, x)

plt.title('Noise and Validation-Accuracy (50 epochs)')
plt.xlabel('Validation-Accuracy',labelpad=10)
plt.ylabel('Features',labelpad=10)
plt.legend(('No Noise', 'Enviromental Noise', 'No Noise and Env. Noise mixed'))
plt.tight_layout()

# Show graphic
plt.show()
