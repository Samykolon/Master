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

m=[]
mmin=[]
mmax=[]

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

with open('data_both_mixednoise2000.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        m.append(float(row[8]))
        mmin.append(float(row[8])-float(row[6]))
        mmax.append(float(row[7])-float(row[8]))

merror = [mmin, mmax]

y_pos = np.arange(len(x))

plt.xlim(18,90)
plt.ylim()
# Create bars
fig1 = plt.barh(y_pos, l, xerr=lerror, height=0.20, label='Mixed noise (4000 samples)')
fig2 = plt.barh(y_pos+0.20, m, xerr=merror, height=0.20, label='Mixed noise (2000 samples)')
fig3 = plt.barh(y_pos+0.40, k, xerr=kerror, height=0.20, label='Enviromental noise')
fig3 = plt.barh(y_pos+0.60, y, xerr=yerror, height=0.20, label='No noise')

# Create names on the x-axis
plt.yticks(y_pos+0.3, x)


plt.title('Noise and Validation-Accuracy (50 epochs)')
plt.xlabel('Validation-Accuracy',labelpad=10)
plt.ylabel('Features',labelpad=10)
plt.legend()
# plt.legend(('Mixed noise (4000 samples)', 'Mixed noise (2000 samples)', 'Enviromental noise', 'No noise'))


ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
plt.tight_layout()
# Show graphic
plt.show()
