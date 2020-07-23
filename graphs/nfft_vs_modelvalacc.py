import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import glob, os
import csv
import numpy as np

x=[]
y=[]

a=[]
b=[]

os.chdir("/home/smu/Desktop/RNN/graphs")

with open('nfft_vs_valacc_1lstm.csv', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y.append(float(row[1]))

with open('nfft_vs_valacc_4lstm.csv', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter=',')
    for row in plots:
        a.append(int(row[0]))
        b.append(float(row[1]))

n = len(x)
n = np.arange(n)

line1, = plt.plot(n,y, marker='o', label="1xLSTM")
line2, = plt.plot(n,b, marker='o', label="4xLSTM")

plt.legend()
plt.xticks(n,x)

plt.title('NFFT-Size and Validation-Accuracy')

plt.xlabel('NFFT-Size')
plt.ylabel('Validation-Accuracy')

plt.show()
