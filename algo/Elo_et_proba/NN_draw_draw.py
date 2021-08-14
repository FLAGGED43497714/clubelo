import numpy as np 
from matplotlib import pyplot as plt
from math import *
from NN_draw import Neural_Network

#Path :
path_csv = "data/NN_egalites\egaliteDeltaElo_v2.csv"

x_1 = np.genfromtxt(path_csv, delimiter=",")[0:, 0]
x_entrer = np.array([[0,1] for k in range(len(x_1))])
for k in range(len(x_1)) :
  x_entrer[k][0] = int(x_1[k])

maxdelta = np.amax(x_entrer, axis=0)[0]

# Changement de l'échelle de nos valeurs pour être entre 0 et 1
x_entrer = x_entrer/np.amax(x_entrer, axis=0) # On divise chaque entré par la valeur max des entrées

# On récupère ce qu'il nous intéresse
X = np.split(x_entrer, [-2])[0] # Données sur lesquelles on va s'entrainer, les 8 premières de notre matrice
xPrediction = np.split(x_entrer, [-1])[1] # Valeur que l'on veut trouver

y_1 = np.genfromtxt(path_csv, delimiter=",")[0:-1, 1]
y_1 = y_1[:-1]

y = np.array([[0] for k in range(len(y_1))])
for k in range(len(y_1)) :
  y[k][0] = int(y_1[k])


NN = Neural_Network()


prob_dens = [0 for k in range(60)]
for k in range(60) :
    prob_dens[k] = NN.forward([5*k / maxdelta, 1])
x_axis = [5*k for k in range(60)]


s = 0
s_quad = 0
for k in range(len(X)) :
  error = NN.forward(X[k]) - y_1[k]
  quad_error = error **2
  s+=error
  s_quad += quad_error

error_moy = s/len(X)
error_quad_moy = s_quad/len(X)

print("erreur")
print(error_moy)
print("quad_error")
print(error_quad_moy)


plt.plot(x_axis, prob_dens)
plt.show()