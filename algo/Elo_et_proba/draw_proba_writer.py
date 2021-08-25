from contextlib import nullcontext
from tokenize import String
import numpy as np
import pandas as pd
from NN_draw import Neural_Network
from matplotlib import pyplot as plt
from egalite_deltaElo_v2 import dataSetDraw
from Elo_and_HFA_v2 import one_match_update
with open("config/k_value.txt") as f :
    K =  int(f.read())

#Path :
def draw_proba_writer(path_matchs, path_start_elo, imported=False, path_import_W1="data/W1.dat", path_import_W2="data/W2.dat", 
  saved=False, path_save_W1="data/W1.dat", path_save_W2="data/W2.dat") :
  res = dataSetDraw(path_matchs, path_start_elo)
  
  ##### Entrainement #####

  x_1 = np.array(res)[0:, 0]
  x_entrer = np.array([[0,1] for k in range(len(x_1))])
  for k in range(len(x_1)) :
    x_entrer[k][0] = int(x_1[k])

  x_entrer_abs = [abs(x_entrer[k][0]) for k in range(len(x_entrer))]
  maxdelta = np.amax(x_entrer_abs)

  # Changement de l'échelle de nos valeurs pour être entre 0 et 1
  x_entrer[k] = [x_entrer[k][0]/maxdelta , 1] # On divise chaque entré par la valeur max des entrées

  # print(maxdelta)
  # print(x_entrer)
  # On récupère ce qu'il nous intéresse
  X = np.split(x_entrer, [-2])[0] # Données sur lesquelles on va s'entrainer, les 8 premières de notre matrice



  y_1 = np.array(res)[0:-1, 1]
  y_1 = y_1[:-1]

  y = np.array([[0] for k in range(len(y_1))])
  for k in range(len(y_1)) :
    y[k][0] = y_1[k]

  # X_neg = np.array([])
  # X_pos = np.array([[None,None]])
  # y_neg = np.array([[None]])
  # y_pos = np.array([[None]])
  # for k in range(len(X)) :
  #   print("X[k][0]")
  #   print(X[k][0])
  #   if X[k][0] < 0 :
  #     print(np.array([X[k]]))
      
  #     np.append(X_neg, np.array([X[k]]), axis=0)
  #     np.append(y_neg, y[k])
  #   else :
  #     np.append(X_pos, [X[k]], axis = 0)
  #     np.append(y_pos, y[k])

  # print("X_neg")
  # print(X_neg)
  # print("X_pos")
  # print(X_pos)
  # print("y_neg")
  # print(y_neg)
  # print("y_pos")
  # print(y_pos)

  if ( imported == False ) :
    # print("NN is created, not imported")
    error_moy = 1
    error_quad_moy = 1
    while(abs(error_moy) > 0.245 and error_quad_moy > 0.245) :
      NN = Neural_Network()
      for i in range(500): #Choisissez un nombre d'itération, attention un trop grand nombre peut créer un overfitting !
        NN.train(X,y)
      s = 0
      s_quad = 0
      for k in range(len(X)) :
        # print(str(X[k]) + " donne "+ str(NN.forward([X[k][0]/maxdelta, 1])))
        error = (NN.forward(X[k]) - y_1[k])
        error_quad = error**2 
        s+=error
        s_quad += error_quad
      error_moy = s/len(X)
      error_quad_moy = s_quad/len(X)
    if ( saved == True ) :
      # print("NN is saved in : "+path_save_W1+" and "+path_save_W2)
      NN.save(out_W1=path_save_W1, out_W2=path_save_W2)
  else :
    # print("NN is imported from : "+path_import_W1+" and "+path_import_W2 + "\n" 
    # + "Using maxdelta = "+str(400.90837938) + " and not : " +str(maxdelta))
    maxdelta = 400.90837938
    print("maxdelta is : "+ str(maxdelta))
    NN = Neural_Network()
    NN.set(from_W1=path_import_W1, from_W2=path_import_W2)

  # prob_dens = [0 for k in range(60)]
  # for k in range(60) :
  #     prob_dens[k] = NN.forward([5*k / maxdelta, 1])
  # x_axis = [5*k for k in range(60)]
  # plt.plot(x_axis, prob_dens)
  # plt.show()


    




  ####### Prédictions #######

  Matchs = np.genfromtxt(path_matchs, delimiter=',', dtype=str)[0:, 0:11]

  nb_de_matchs = len(Matchs)

  Matchs_predictions_names = np.array([[Matchs[k][0], Matchs[k][1]] for k in range(nb_de_matchs)], dtype=str) 
  Matchs_predictions_nbrs = np.array([[0, 0, 0, 0] for k in range(nb_de_matchs)], dtype=float)


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

    absDeltaElo = abs(homeElo - awayElo)
    deltaElo = homeElo - awayElo
    Matchs_predictions_nbrs[match_nb][0] = deltaElo

    draw_percentage = NN.forward([deltaElo/maxdelta, 1])[0]
    Matchs_predictions_nbrs[match_nb][2] = draw_percentage
    currentElo_home2, currentElo_away2 = one_match_update(Matchs, currentElo_home2, currentElo_away2, match_nb, K)

  # df_elo_away = pd.DataFrame(Matchs_predictions)
  # df_elo_away.to_csv("test101", index=False, header = False)
  return Matchs_predictions_names, Matchs_predictions_nbrs

  # prob_dens = [0 for k in range(60)]
  # for k in range(60) :
  #     prob_dens[k] = NN.forward([5*k / maxdelta, 1])
  # x_axis = [5*k for k in range(60)]
  # plt.plot(x_axis, prob_dens)
  # plt.show()