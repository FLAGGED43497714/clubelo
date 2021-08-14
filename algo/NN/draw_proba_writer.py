import numpy as np
import pandas as pd
from NN_draw_draw import Neural_Network

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

for i in range(150): #Choisissez un nombre d'itération, attention un trop grand nombre peut créer un overfitting !
    NN.train(X,y)


#######

