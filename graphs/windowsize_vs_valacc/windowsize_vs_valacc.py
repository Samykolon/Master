import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import glob, os
import csv
import numpy as np
from scipy.interpolate import interp1d

plt.rcParams.update({'font.size': 14})

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

f = interp1d(x, y, fill_value="extrapolate")
f2 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
xnew = np.linspace(0.0, 1.6, num=41, endpoint=True)
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')

plt.legend(['data', 'linear', 'cubic'], loc='best')




plt.title('Window-Size and Validation-Accuracy (50 epochs)')
plt.xlabel('Window-Size',labelpad=10)
plt.ylabel('Validation-Accuracy',labelpad=10)
plt.tight_layout()

# Show graphic
plt.show()

# 56.7364 + 37.7661 x - 31.2855 x^2 + 7.53548 x^3
