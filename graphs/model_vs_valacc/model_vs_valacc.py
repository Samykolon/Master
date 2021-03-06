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

os.chdir("/home/smu/Desktop/RNN/graphs/model_vs_valacc")

with open('model_vs_valacc.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        x.append(str(row[0]))
        y.append(float(row[7]))

y_pos = np.arange(len(x))

fig, ax = plt.subplots()

plt.xlim(40,85)
# Create bars
plt.barh(y_pos, y)

# Create names on the x-axis
plt.yticks(y_pos, x)

for i, v in enumerate(y):
    ax.text(v + 1.5, i-0.1, str(v))

plt.xlabel('Validations-Genauigkeit',labelpad=10)
plt.ylabel('Netzstruktur',labelpad=10)
plt.tight_layout()

# Show graphic
plt.savefig('model_vs_valacc.pgf')
