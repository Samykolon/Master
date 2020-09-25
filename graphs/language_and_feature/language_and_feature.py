import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import glob, os
import csv
import numpy as np

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 6
fig_size[1] = 5

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
ymin=[]
ymax=[]

k=[]
kmin=[]
kmax=[]

l=[]
lmin=[]
lmax=[]

os.chdir("/home/smu/Desktop/RNN/graphs/language_and_feature")

with open('data_both_valacc.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        x.append(str(row[0]))
        ymin.append(float(row[8])-float(row[6]))
        ymax.append(float(row[7])-float(row[8]))
        y.append(float(row[8]))

yerror = [ymin, ymax]

with open('data_ger_valacc.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        k.append(float(row[8]))
        kmin.append(float(row[8])-float(row[6]))
        kmax.append(float(row[7])-float(row[8]))

kerror = [kmin, kmax]

with open('data_eng_valacc.csv', 'r') as csvfile:
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
fig1 = plt.barh(y_pos, y, xerr=yerror, height=0.25, label='Deutsch + Englisch')
fig2 = plt.barh(y_pos+0.25, k, xerr=kerror, height=0.25, label='Deutsch')
fig3 = plt.barh(y_pos+0.5, l, xerr=lerror, height=0.25, label='Englisch')

# Create names on the x-axis
plt.yticks(y_pos+0.25, x)

plt.xlabel('Validations-Genauigkeit',labelpad=10)

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
plt.tight_layout()

# Show graphic
plt.savefig('language_and_feature.pgf')
