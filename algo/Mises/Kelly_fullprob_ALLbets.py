import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def mise(C, K, p) :
    if K == 0 :
        return 0
    if K*p < 1 :
        return 0
    f = p - (1 - p) / ( K - 1 )
    return C * f 

path_csv = "data\dataBacktest\\2020_2021_test1.csv"


path_out = 'data/testfile1.csv'
#K pour les cotes, p pour les probas
#W = win, D = draw, L = loose

K_array_W = np.genfromtxt(path_csv, delimiter=",")[4:, 0]
p_array_W = np.genfromtxt(path_csv, delimiter=",")[4:, 1]

K_array_D = np.genfromtxt(path_csv, delimiter=",")[4:, 4]
p_array_D = np.genfromtxt(path_csv, delimiter=",")[4:, 5]

K_array_L = np.genfromtxt(path_csv, delimiter=",")[4:, 8]
p_array_L = np.genfromtxt(path_csv, delimiter=",")[4:, 9]

winner_array = np.genfromtxt(path_csv, delimiter=",")[4:, 11]


nb_match = len(K_array_W)



#Espérance des paris 
G_array_W = np.array([K_array_W[k] * p_array_W[k] for k in range(nb_match)])
G_array_D = np.array([K_array_D[k] * p_array_D[k] for k in range(nb_match)])
G_array_L = np.array([K_array_L[k] * p_array_L[k] for k in range(nb_match)])


def bestBet(esp1, esp2, esp3) :
    if max(esp1, esp2, esp3) >= 1.5 :
        return -1
    return max(esp1, esp2, esp3)


allbets = np.array([[0, 0, 0, 0]])
minEsp = 1.1
maxEsp = 1.2
for k in range(nb_match) :
    if minEsp < G_array_W[k] < maxEsp :
        # print([ G_array_W[k], K_array_W[k], p_array_W[k], 1 ])
        allbets = np.append(allbets, [[ G_array_W[k], K_array_W[k], p_array_W[k], 1 ]], axis=0 )
        # print(allbets)
    if minEsp < G_array_D[k] < maxEsp :
        allbets = np.append(allbets, [[G_array_D[k], K_array_D[k], p_array_D[k], 2]], axis=0 )
    if minEsp < G_array_L[k] < maxEsp :
        allbets = np.append(allbets, [[G_array_L[k], K_array_L[k], p_array_L[k], 3]], axis=0 )


#Paris choisis [[G,K,p]]

#C = Capital 
C_array = np.array([[100]], dtype=float)

for i in range(len(allbets)) : 

    s = mise(C_array[i], allbets[i][1], allbets[i][2])
    if (allbets[i][3] == winner_array[i]) :
        res = 1 
    else :
        res = 0
    C_array = np.append(C_array, [C_array[i] -s + s * (res) * allbets[i][1]], axis=0)

df = pd.DataFrame(C_array)
df.to_csv(path_out, index=False, header = False)
