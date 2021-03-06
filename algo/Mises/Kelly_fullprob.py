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

path_csv = "data\dataBacktest\\2020_2021_test22.csv"


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
    if (1 > max(esp1, esp2, esp3)) or ( max(esp1, esp2, esp3) > 1.5) :
        return -1
    return max(esp1, esp2, esp3)

#Meilleur des 3 paris
best_G_array = np.array([bestBet(G_array_W[k],G_array_D[k], G_array_L[k]) for k in range(nb_match)])

#Paris choisis [[G,K,p]]
bets_taken = np.array([[best_G_array[k], 0, 0, 0 ] for k in range(nb_match)])
for k in range(nb_match) :
    if (bets_taken[k][0] == G_array_W[k]) :
        bets_taken[k][1] = K_array_W[k]
        bets_taken[k][2] = p_array_W[k]
        bets_taken[k][3] = 1

    if (bets_taken[k][0] == G_array_D[k]) :
        bets_taken[k][1] = K_array_D[k]
        bets_taken[k][2] = p_array_D[k]
        bets_taken[k][3] = 2

    if (bets_taken[k][0] == G_array_L[k]) :
        bets_taken[k][1] = K_array_L[k]
        bets_taken[k][2] = p_array_L[k]
        bets_taken[k][3] = 3

#C = Capital 
C_array = np.array([100 for k in range(nb_match+1)], dtype=float)

for i in range(nb_match) : 
    s = mise(C_array[i], bets_taken[i][1], bets_taken[i][2])
    if (bets_taken[i][3] == winner_array[i]) :
        res = 1 
    else :
        res = 0
    C_array[i+1] = C_array[i] -s + s * (res) * bets_taken[i][1]

df = pd.DataFrame(C_array)
df.to_csv(path_out, index=False, header = False)
