import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def esperance_morale(C,M,K,p) :
    # print("in esperance morale :")
    # print("C = ")
    # print(C)
    # print("M = ")
    # print(M)
    # print("K = ")
    # print(K)
    # print("p = ")
    # print(p)


    r = C * ( ( (M*K+C) / C )**p * ( (-M+C) / C )**(1 - p) )
    return r

x = [ ( 1 + m/10 )  for m in range (100)]

def mise(C, x, K, p) :
    y = np.amax(np.array([esperance_morale(C, x[k], K, p) for k in range(len(x))])) - C
    if y < 0 :
        y = 0
    return y

# print("3.6 0.32")
# print(mise(89.2169, x, 3.6, 0.32))





path_csv = "data\Paris_2021.csv"

#K pour les cotes, p pour les probas
#W = win, D = draw, L = loose

K_array_W = np.genfromtxt(path_csv, delimiter=",")[4:, 0]
p_array_W = np.genfromtxt(path_csv, delimiter=",")[4:, 1]

K_array_D = np.genfromtxt(path_csv, delimiter=",")[4:, 4]
p_array_D = np.genfromtxt(path_csv, delimiter=",")[4:, 5]

K_array_L = np.genfromtxt(path_csv, delimiter=",")[4:, 8]
p_array_L = np.genfromtxt(path_csv, delimiter=",")[4:, 9]

nb_match = len(K_array_W)

#results (0 si le paris est raté, 1 si il est réussi)
#A changer
res = np.genfromtxt(path_csv, delimiter=",")[4:, 13]


#Espérance des paris 
G_array_W = np.array([K_array_W[k] * p_array_W[k] for k in range(nb_match)])
G_array_D = np.array([K_array_D[k] * p_array_D[k] for k in range(nb_match)])
G_array_L = np.array([K_array_L[k] * p_array_L[k] for k in range(nb_match)])

#Meilleur des 3 paris
best_G_array = np.array([max(G_array_W[k],G_array_D[k], G_array_L[k]) for k in range(nb_match)])

#Paris choisis [[G,K,p]]
bets_taken = np.array([[best_G_array[k], 0, 0 ] for k in range(nb_match)])
for k in range(nb_match) :
    if (bets_taken[k][0] == G_array_W[k]) :
        bets_taken[k][1] = K_array_W[k]
        bets_taken[k][2] = p_array_W[k]

    if (bets_taken[k][0] == G_array_D[k]) :
        bets_taken[k][1] = K_array_D[k]
        bets_taken[k][2] = p_array_D[k]

    if (bets_taken[k][0] == G_array_L[k]) :
        bets_taken[k][1] = K_array_L[k]
        bets_taken[k][2] = p_array_L[k]


#C = Capital 
C_array = np.array([100 for k in range(nb_match+1)], dtype=float)

for i in range(nb_match) : 
    s = mise(C_array[i], x, bets_taken[i][1], bets_taken[i][2])
    C_array[i+1] = C_array[i] -s + s * res[i] * bets_taken[i][1]

df = pd.DataFrame(C_array)
df.to_csv('data/testfile.csv')


# y = [np.amax(np.array([esperance_morale(100, x[k], K_array[i], p_array[i]) for k in range(len(x))])) - 100 for i in range(len(p_array))]
# print(y)

# df = pd.DataFrame(y)
# df.to_csv('data/testfile.csv')


# plt.plot(x,y)
# plt.show()