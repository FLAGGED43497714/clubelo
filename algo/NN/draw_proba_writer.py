import numpy as np
import pandas as pd
from NN_draw import Neural_Network
from matplotlib import pyplot as plt
import sys
sys.path[0] = 'c:\\Users\\balth\\python-workspace\\clubelo\\algo\\Elo_et_proba'
from egalite_deltaElo_v2 import dataSetDraw
from Elo_and_HFA_v2 import one_match_update
with open("config/k_value.txt") as f :
    K =  int(f.read())

#Path :

path_matchs = "data\saisons\saison_2021_sans_cotes.csv"
path_start_elo = "data\elo_start\elo_start_2020_v3.csv"

res = dataSetDraw(path_matchs, path_start_elo)

##### Entrainement #####

x_1 = np.array(res)[0:, 0]
x_entrer = np.array([[0,1] for k in range(len(x_1))])
for k in range(len(x_1)) :
  x_entrer[k][0] = int(x_1[k])

maxdelta = np.amax(x_entrer, axis=0)[0]

# Changement de l'échelle de nos valeurs pour être entre 0 et 1
x_entrer = x_entrer/np.amax(x_entrer, axis=0) # On divise chaque entré par la valeur max des entrées

# On récupère ce qu'il nous intéresse
X = np.split(x_entrer, [-2])[0] # Données sur lesquelles on va s'entrainer, les 8 premières de notre matrice



y_1 = np.array(res)[0:-1, 1]
y_1 = y_1[:-1]

y = np.array([[0] for k in range(len(y_1))])
for k in range(len(y_1)) :
  y[k][0] = y_1[k]




error_quad_moy = 1
while(error_quad_moy > 0.245) :
  NN = Neural_Network()
  for i in range(200): #Choisissez un nombre d'itération, attention un trop grand nombre peut créer un overfitting !
    NN.train(X,y)
  s = 0
  for k in range(len(X)) :
    error_quad = (NN.forward(X[k]/maxdelta) - y_1[k])**2
    s+=error_quad
  error_quad_moy = s/len(X)
  print(s/len(X))

  



####### Prédictions

Matchs = np.genfromtxt(path_matchs, delimiter=',', dtype=str)[0:, 0:4]

nb_de_matchs = len(Matchs)

Matchs_predictions = np.array([[Matchs[k][0], Matchs[k][1], 0, 0.0, 0.0, 0.0] for k in range(nb_de_matchs)])


Elo_name = np.genfromtxt(path_start_elo, delimiter=',', dtype=str)[1:, 0]
Elo_home = np.genfromtxt(path_start_elo, delimiter=',', dtype=str)[1:, 1]
Elo_away = np.genfromtxt(path_start_elo, delimiter=',', dtype=str)[1:, 2]

nb_d_equipe = len(Elo_name)

currentElo_home2 = [[Elo_name[k],Elo_home[k]] for k in range(nb_d_equipe)]
currentElo_away2 = [[Elo_name[k],Elo_away[k]] for k in range(nb_d_equipe)]

for match_nb in range(nb_de_matchs) :
  homeTeam = Matchs[match_nb][0]
  awayTeam = Matchs[match_nb][1]
  for k in range(nb_d_equipe) :
    if currentElo_home2[k][0] == homeTeam :
        homeElo = float(currentElo_home2[k][1])
    if currentElo_away2[k][0] == awayTeam :
        awayElo = float(currentElo_away2[k][1])

  deltaElo = homeElo - awayElo
  Matchs_predictions[match_nb][2] = deltaElo

  draw_percentage = NN.forward([deltaElo/maxdelta, 1])[0]

  Matchs_predictions[match_nb][4] = draw_percentage
  currentElo_home2, currentElo_away2 = one_match_update(Matchs, currentElo_home2, currentElo_away2, match_nb, K)

  df = pd.DataFrame(Matchs_predictions)
  df.to_csv('data/testfile.csv')

prob_dens = [0 for k in range(60)]
for k in range(60) :
    prob_dens[k] = NN.forward([5*k / maxdelta, 1])
x_axis = [5*k for k in range(60)]
plt.plot(x_axis, prob_dens)
plt.show()