import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


import glob, os
import csv
import numpy as np
from scipy.interpolate import interp1d

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

os.chdir("/home/smu/Desktop/RNN/graphs/windowsize_vs_valacc")

with open('windowsize_vs_valacc.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[7]))

plt.ylim(50,80)
# plt.plot(x, y, marker='o')

f2 = interp1d(x, y, kind='cubic', fill_value="extrapolate", )
xnew = np.linspace(0.0, 1.6, num=41, endpoint=True)
plt.plot(x, y, 'o', xnew, f2(xnew), '--', color='#3268a8', markersize=10)

for i,j in zip(x,y):
    if i == 0.05:
        plt.annotate(str(j),xy=(i+0.06,j-0.7))
    else:
        plt.annotate(str(j),xy=(i-0.04,j+2.5))

plt.xlabel('Fenstergröße',labelpad=10)
plt.ylabel('Validations-Genauigkeit',labelpad=10)
plt.tight_layout()

# Show graphic
plt.savefig('windowsize_vs_valacc.pgf')

# 56.7364 + 37.7661 x - 31.2855 x^2 + 7.53548 x^3
