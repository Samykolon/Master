import matplotlib.pyplot as plt
import numpy as np
import math

# Create the vectors X and Y
x = np.array(range(10000))
tempx = x / 700
tempx = 1 + tempx
y = 2595 * np.log10(tempx)

print(y)

# Create the plot
plt.plot(x,y)

plt.xlabel('Frequenzskala in Hz')
plt.ylabel('Melskala in Mel')
plt.grid(alpha=.6,linestyle='--')


# Show the plot
plt.show()
