import numpy as np
from draw_proba_writer import draw_proba_writer
from Score_estimation1 import score_estimation2
import pandas as pd

path_matchs = "data\saisons\\2020_2021_avec_v4.csv"
path_start_elo = "data\elo_start\elo_start_2020_v3.csv"
path_out_prob = "data\probs.csv"

realistic = True

# match_prediction = draw_proba_writer(path_matchs, path_start_elo, imported=True, 
# path_import_W1="data/W1_test3.dat", path_import_W2="data/W2_test3.dat")
# maxCap = 0
# while(maxCap < 1000 or realistic == False) :
match_prediction = draw_proba_writer(path_matchs, path_start_elo, saved=True, 
path_save_W1="data/W1_allbets1.dat", path_save_W2="data/W2_allbets1.dat")

nb_of_matchs = len(match_prediction)

for match_nb in range(nb_of_matchs) :
    delta_elo = float(match_prediction[match_nb][2])
    perct_restant = 1 - float(match_prediction[match_nb][4])
    # print(match_prediction)
    # print("perct_restant")
    # print(perct_restant)
    Eh = score_estimation2(delta_elo)

    home_percent = float(( 1 - Eh ) * perct_restant)
    away_percent = float(Eh * perct_restant)
    match_prediction[match_nb][3] = float(home_percent)
    match_prediction[match_nb][5] = float(away_percent)

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
    p_array_W[match_nb] = match_prediction[match_nb][3]
    p_array_D[match_nb] = match_prediction[match_nb][4]
    p_array_L[match_nb] = match_prediction[match_nb][5]

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
minEsp = 1.3
maxEsp = 1.4
for k in range(nb_match) :
    if minEsp < G_array_W[k] < maxEsp :
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
print(str(realistic))
if (realistic) :
    print("Benefices finaux : "+str(finCap-100))
    print("Benefices max : "+str(maxCap-100))

df = pd.DataFrame(C_array)
df.to_csv(path_out, index=False, header = False)
