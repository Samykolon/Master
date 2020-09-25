import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import glob, os
import csv
import numpy as np

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 6
fig_size[1] = 3

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

os.chdir("/home/smu/Desktop/RNN/graphs/mfcc_vs_valacc")

with open('mfcc_vs_valacc.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        x.append(str(row[0]))
        y.append(float(row[7]))

y_pos = np.arange(len(x))

plt.ylim(65,85)
# Create bars
plt.bar(y_pos, y)

# Create names on the x-axis
plt.xticks(y_pos, x)

for a,b in zip(y_pos,y):

    plt.annotate(b, # this is the text
                 (a,b), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('Anzahl der Mel-Frequenz-Cepstrum-Koeffizienten',labelpad=10)
plt.ylabel('Validations-Genauigkeit',labelpad=10)
plt.tight_layout()

# Show graphic
plt.savefig('mfcc_vs_valacc.pgf')
