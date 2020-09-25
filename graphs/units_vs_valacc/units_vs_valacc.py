import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import glob, os
import csv
import numpy as np

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 6
fig_size[1] = 4

#plt.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'figure.figsize': fig_size,

})

x=[]
y=[]
k=[]

os.chdir("/home/smu/Desktop/RNN/graphs/units_vs_valacc")

with open('units_vs_valacc_5lstm.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        x.append(str(row[0]))
        y.append(float(row[7]))

with open('units_vs_valacc_resnet.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        k.append(float(row[7]))

y_pos = np.arange(len(x))

plt.ylim(45,80)
# Create bars

fig1 = plt.bar(y_pos, y, width=0.25)
fig2 = plt.bar(y_pos+0.25, k, width=0.25)

# Create names on the x-axis
plt.xticks(y_pos+0.125, x)

plt.xlabel('Anzahl der LSTM-Zellen pro Schicht',labelpad=10)
plt.ylabel('Validationsgenauigkeit',labelpad=10)
plt.legend(('5LSTM', 'RESNET'))
plt.tight_layout()

# Show graphic
plt.savefig('units_vs_valacc.pgf')
