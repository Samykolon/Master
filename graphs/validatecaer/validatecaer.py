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

l=[]

m=[]

os.chdir("/home/smu/Desktop/RNN/graphs/validatecaer")

with open('nonoise.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        x.append(str(row[0]))
        y.append(float(row[1]))



with open('envnoise.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        k.append(float(row[1]))

with open('mixednoise4000.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        l.append(float(row[1]))


with open('mixednoise2000.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        m.append(float(row[1]))

y_pos = np.arange(len(x))

plt.xlim(0,30)
plt.ylim()
# Create bars
fig1 = plt.barh(y_pos, l, height=0.20, label='Mixed noise (4000 samples)')
fig2 = plt.barh(y_pos+0.20, m, height=0.20, label='Mixed noise (2000 samples)')
fig3 = plt.barh(y_pos+0.40, k, height=0.20, label='Enviromental noise')
fig3 = plt.barh(y_pos+0.60, y, height=0.20, label='No noise')

# Create names on the x-axis
plt.yticks(y_pos+0.3, x)


plt.xlabel('Validations-Genauigkeit',labelpad=10)
plt.legend()


ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc=3, bbox_to_anchor=(0.6, 0.63), prop={'size': 6})
plt.tight_layout()
# Show graphic
#plt.show()
plt.savefig('validatecaer.pgf')
