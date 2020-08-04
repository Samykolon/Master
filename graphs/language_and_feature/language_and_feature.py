import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import glob, os
import csv
import numpy as np

x=[]
y=[]
k=[]
l=[]

os.chdir("/home/smu/Desktop/RNN/graphs/language_and_feature")

with open('data_both_valacc.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        x.append(str(row[0]))
        y.append(float(row[7]))

with open('data_ger_valacc.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        k.append(float(row[7]))

with open('data_eng_valacc.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        l.append(float(row[7]))

y_pos = np.arange(len(x))

plt.xlim(25,85)
# Create bars
fig1 = plt.barh(y_pos, y, height=0.25)
fig2 = plt.barh(y_pos+0.25, k, height=0.25)
fig3 = plt.barh(y_pos+0.5, l, height=0.25)

# Create names on the x-axis
plt.yticks(y_pos+0.25, x)

plt.title('Features and Validation-Accuracy (50 epochs)')
plt.xlabel('Validation-Accuracy',labelpad=10)
plt.ylabel('Features',labelpad=10)
plt.legend(('Both', 'Ger', 'Eng'))
plt.tight_layout()

# Show graphic
plt.show()
