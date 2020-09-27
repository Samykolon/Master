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

m=[]
mmin=[]
mmax=[]

os.chdir("/home/smu/Desktop/RNN/graphs/validatetrained")

with open('nonoise.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        x.append(str(row[0]))
        ty = float(row[1])
        tymax = float(row[2])
        if (tymax >= ty):
            ymin.append(float(row[1]))
            ymax.append(float(tymax-ty))
        else:
            ymin.append(float(row[1]))
            ymax.append(float(tymax-ty))

        y.append(float(row[1]))

yerror = [ymin, ymax]



with open('envnoise.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        tk = float(row[1])
        tkmax = float(row[2])
        if (tkmax >= tk):
            kmin.append(float(row[1]))
            kmax.append(float(tkmax-tk))
        else:
            kmin.append(float(row[1]))
            kmax.append(float(tkmax-tk))

        k.append(float(row[1]))

kerror = [kmin, kmax]

with open('mixednoise4000.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        tl = float(row[1])
        tlmax = float(row[2])
        if (tlmax >= tl):
            lmin.append(float(row[1]))
            lmax.append(float(tlmax-tl))
        else:
            lmin.append(float(row[1]))
            lmax.append(float(tlmax-tl))

        l.append(float(row[1]))

lerror = [lmin, lmax]


with open('mixednoise2000.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        tm = float(row[1])
        tmmax = float(row[2])
        if (tmmax >= tm):
            mmin.append(float(row[1]))
            mmax.append(float(tmmax-tm))
        else:
            mmin.append(float(row[1]))
            mmax.append(float(tmmax-tm))

        m.append(float(row[1]))

merror = [mmin, mmax]

y_pos = np.arange(len(x))

plt.xlim(10,90)
plt.ylim()
# Create bars
fig1 = plt.barh(y_pos, l, xerr=lerror, height=0.20, label='Mixed noise (4000 samples)')
fig2 = plt.barh(y_pos+0.20, m, xerr=merror, height=0.20, label='Mixed noise (2000 samples)')
fig3 = plt.barh(y_pos+0.40, k, xerr=kerror, height=0.20, label='Enviromental noise')
fig3 = plt.barh(y_pos+0.60, y, xerr=yerror, height=0.20, label='No noise')

# Create names on the x-axis
plt.yticks(y_pos+0.3, x)


plt.xlabel('Validations-Genauigkeit',labelpad=10)
plt.legend()

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], prop={'size': 6})
plt.tight_layout()
# Show graphic
plt.savefig('validatetrained.pgf')
