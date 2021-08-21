import numpy as np
from matplotlib import pyplot as plt

path_c0 = "data\dataBacktest\loi_normale_v2.csv"

c0 = np.genfromtxt(path_c0, delimiter=",", dtype=float)

nb_of_bets = len(c0) 

norm = [0 for k in range(100)]
x = [-0.505 +j*0.01 for j in range(100)]

for k in range(nb_of_bets) :
    for j in range(100) :
        if (-0.5 + j * 0.01 < c0[k] ) and (c0[k] < -0.5 + (j+1) * 0.01 ) :
                norm[j] += 1 

plt.scatter(x,norm)
plt.show()