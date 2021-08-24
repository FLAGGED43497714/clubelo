import numpy as np
from draw_proba_writer import draw_proba_writer
from Score_estimation1 import score_estimation2
import pandas as pd
import math

path_matchs = "data\saisons\\2020_2021_avec_v4.csv"
path_start_elo = "data\elo_start\elo_start_2020_v3.csv"
path_out_prob = "data\probs.csv"
# path_matchs = "data\saisons\\2020_2021_avec_PL_v1.csv"
# path_start_elo = "data\elo_start\elo_start_PL_2020.csv"


realistic = True

# match_prediction = draw_proba_writer(path_matchs, path_start_elo, imported=True, 
# path_import_W1="data/W1_test3.dat", path_import_W2="data/W2_test3.dat")
# maxCap = 0
# while(maxCap < 1000 or realistic == False) :
match_prediction_names, match_prediction_nbrs  = draw_proba_writer(path_matchs, path_start_elo, imported=True, 
path_import_W1="data/NN_saved/W1_BR100n1", path_import_W2="data/NN_saved/W2_BR100n1")

nb_of_matchs = len(match_prediction_names)

for match_nb in range(nb_of_matchs) :
    delta_elo = float(match_prediction_nbrs[match_nb][0])
    perct_restant = 1 - float(match_prediction_nbrs[match_nb][2])
    # print(match_prediction)
    # print("perct_restant")
    # print(perct_restant)
    Eh = score_estimation2(delta_elo)

    home_percent = float(( 1 - Eh ) * perct_restant)
    away_percent = float(Eh * perct_restant)
    match_prediction_nbrs[match_nb][1] = float(home_percent)
    match_prediction_nbrs[match_nb][3] = float(away_percent)

# df = pd.DataFrame(match_prediction)



def mise(C, K, p) :
    if K == 0 :
        return 0
    if K*p < 1 :
        return 0
    f = p - (1 - p) / ( K - 1 )
    return C * f 



path_out = 'data/testfile1.csv'
#K pour les cotes, p pour les probas
#W = win, D = draw, L = loose

# Ce qui ne change pas 
K_array_W = np.genfromtxt(path_matchs, delimiter=",")[:, 7]
K_array_D = np.genfromtxt(path_matchs, delimiter=",")[:, 8]
K_array_L = np.genfromtxt(path_matchs, delimiter=",")[:, 9]
winner_array = np.genfromtxt(path_matchs, delimiter=",")[:, 10]


# Ce qui dépend du NN
p_array_W = np.array([0 for match_nb in range(nb_of_matchs)], dtype=float)
p_array_D = np.array([0 for match_nb in range(nb_of_matchs)], dtype=float)
p_array_L = np.array([0 for match_nb in range(nb_of_matchs)], dtype=float)
for match_nb in range(nb_of_matchs) :
    p_array_W[match_nb] = match_prediction_nbrs[match_nb][1]
    p_array_D[match_nb] = match_prediction_nbrs[match_nb][2]
    p_array_L[match_nb] = match_prediction_nbrs[match_nb][3]

if np.amax(p_array_W) > 1 :
    realistic = False

if np.amax(p_array_D) > 1 :
    realistic = False

if np.amax(p_array_L) > 1 :
    realistic = False


nb_match = len(K_array_W)



#Espérance des paris 
G_array_W = np.array([K_array_W[k] * p_array_W[k] for k in range(nb_match)])
G_array_D = np.array([K_array_D[k] * p_array_D[k] for k in range(nb_match)])
G_array_L = np.array([K_array_L[k] * p_array_L[k] for k in range(nb_match)])

preds = np.array([[p_array_W[k],p_array_D[k],p_array_L[k], G_array_W[k], G_array_D[k], G_array_L[k]
] for k in range(nb_of_matchs) ], dtype=float)
df = pd.DataFrame(preds)
df.to_csv(path_out_prob, index=False, header = False)



allbets = np.array([[0, 0, 0, 0, 0]])
minEsp = 1
maxEsp = 1.5
threeBetMax = 1.5
for k in range(nb_match) :
    if (max(G_array_W[k], G_array_D[k], G_array_L[k]) < threeBetMax) :
        if (minEsp < G_array_W[k] < maxEsp )  :
            # print([ G_array_W[k], K_array_W[k], p_array_W[k], 1 ])
            allbets = np.append(allbets, [[ G_array_W[k], K_array_W[k], p_array_W[k], 1, k]], axis=0 )
            # print(allbets)
        if minEsp < G_array_D[k] < maxEsp :
            allbets = np.append(allbets, [[G_array_D[k], K_array_D[k], p_array_D[k], 2, k]], axis=0 )
        if minEsp < G_array_L[k] < maxEsp :
            allbets = np.append(allbets, [[G_array_L[k], K_array_L[k], p_array_L[k], 3, k]], axis=0 )



C_array = np.array([[100]], dtype=float)

for i in range(len(allbets)) : 

    s = mise(C_array[i], allbets[i][1], allbets[i][2])

    if (allbets[i][3] == winner_array[int(allbets[i][4])]) :
        res = 1 
    else :
        res = 0
    C_array = np.append(C_array, [C_array[i] -s + s * (res) * allbets[i][1]], axis=0)


maxCap = np.amax(C_array)
finCap = C_array[-1]
log_C = np.array([math.log(C_array[k]) for k in range(len(C_array))])
avg_log = np.mean(log_C)
if np.amin(C_array < 0) :
    avg_log = -10000    

print("realistic = " +str(realistic))
if (realistic) :
    print("Benefices finaux : "+str(finCap-100))
    print("Benefices max : "+str(maxCap-100))
    print("avg_log mean = " + str(avg_log))

df = pd.DataFrame(C_array)
df.to_csv(path_out, index=False, header = False)
