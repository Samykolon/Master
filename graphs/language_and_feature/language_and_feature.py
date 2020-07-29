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

plt.ylim(25,85)
# Create bars
fig1 = plt.bar(y_pos, y, width=0.25)
fig2 = plt.bar(y_pos+0.25, k, width=0.25)
fig3 = plt.bar(y_pos+0.5, l, width=0.25)

# Create names on the x-axis
plt.xticks(y_pos+0.25, x)

# for a,b in zip(y_pos,y):
#
#     plt.annotate(b, # this is the text
#                  (a,b), # this is the point to label
#                  textcoords="offset points", # how to position the text
#                  xytext=(0,10), # distance from text to points (x,y)
#                  ha='center') # horizontal alignment can be left, right or center

plt.title('Features and Validation-Accuracy (50 epochs)')
plt.xlabel('Features',labelpad=10)
plt.ylabel('Validation-Accuracy',labelpad=10)
plt.legend(('Both', 'Ger', 'Eng'))
plt.tight_layout()

# Show graphic
plt.show()
